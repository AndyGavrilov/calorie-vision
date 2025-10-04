"""Утилиты для обучения модели предсказания калорийности блюд"""

import os
import random
import time
from functools import partial
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import timm
import torchmetrics
import yaml
from tqdm import tqdm

from scripts.dataset import create_data_loaders


def seed_everything(seed: int = 42) -> None:
    """
    Устанавливает seed для всех библиотек для воспроизводимости.
    
    Args:
        seed: Значение seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CaloriePredictor(nn.Module):
    """Мультимодальная модель для предсказания калорийности блюд.
    
    Архитектура включает:
    - Vision encoder (EfficientNet-B3) для обработки изображений
    - Text encoder (DistilBERT) для обработки списка ингредиентов
    - Fusion network для объединения модальностей
    - Regression head для предсказания калорий
    """
    
    def __init__(
        self,
        vision_model_name: str = "efficientnet_b3",
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            vision_model_name: Название модели для обработки изображений
            text_model_name: Название модели для обработки текста
            hidden_dim: Размерность скрытых слоев
            dropout_rate: Коэффициент dropout
        """
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = timm.create_model(
            vision_model_name,
            pretrained=True,
            num_classes=0,  # Без классификационной головы
            global_pool='avg'
        )
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Feature dimensions
        vision_dim = self.vision_encoder.num_features
        text_dim = self.text_encoder.config.hidden_size
        
        # Projection layers
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion and regression layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim)
        )
        
        # MLP для массового признака
        self.mass_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3)
        )

        # Расширенный регрессор с учетом массы
        fusion_dim = hidden_dim + hidden_dim + 32  # vision + text + mass
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов нового слоев."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        mass_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass модели с поддержкой массового признака.
        
        Args:
            images: Тензор изображений [batch_size, channels, height, width]
            text_input_ids: Токены текста [batch_size, max_length]
            text_attention_mask: Маска внимания [batch_size, max_length]
            mass_features: Лог-нормализованные массы [batch_size]
            
        Returns:
            Предсказания калорий на 100г [batch_size]
        """
        # Vision processing
        vision_features = self.vision_encoder(images)  # [batch_size, vision_dim]
        vision_proj = self.vision_proj(vision_features)  # [batch_size, hidden_dim]
        
        # Text processing
        text_output = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]  # [batch_size, text_dim]
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]
        
        # Mass processing
        # [batch_size, 1] -> [batch_size, 32]
        mass_emb = self.mass_mlp(mass_features.unsqueeze(1))

        # Fusion всех компонентов
        # [batch_size, hidden_dim * 2 + 32]
        fused = torch.cat([vision_proj, text_proj, mass_emb], dim=1)
        
        # Regression
        calories_pred_per_100g = self.regressor(fused)  # [batch_size, 1]
        
        return calories_pred_per_100g.squeeze(-1)  # [batch_size]


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Создает оптимизатор и планировщик learning rate.
    
    Args:
        model: Модель для обучения
        config: Конфигурация с параметрами обучения
        
    Returns:
        Tuple с оптимизатором и планировщиком learning rate
    """
    # Разные learning rates для разных компонентов
    vision_params = [p for n, p in model.named_parameters() if 'vision_encoder' in n]
    text_params = [p for n, p in model.named_parameters() if 'text_encoder' in n]
    other_params = [p for n, p in model.named_parameters() 
                   if 'vision_encoder' not in n and 'text_encoder' not in n]
    
    param_groups = [
        {'params': vision_params, 'lr': config['vision_lr'], 'name': 'vision'},
        {'params': text_params, 'lr': config['text_lr'], 'name': 'text'},
        {'params': other_params, 'lr': config['fusion_lr'], 'name': 'fusion'}
    ]
    
    optimizer = AdamW(param_groups, weight_decay=config['weight_decay'])
    
    # Новый планировщик: warmup + cosine annealing без рестартов
    import math
    warmup_epochs = config.get('warmup_epochs', 5)
    min_lr = config.get('min_lr', 1e-6)
    total_epochs = config['epochs']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Линейное соотнесение к косинусу без рестартов
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))  # от 1 до 0

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Dict
) -> Tuple[float, float]:
    """
    Обучает модель на одной эпохе.
    
    Args:
        model: Модель для обучения
        train_loader: Загрузчик данных для обучения
        optimizer: Оптимизатор
        criterion: Функция потерь
        device: Устройство для вычислений
        config: Конфигурация
        
    Returns:
        Tuple с средней loss и MAE для эпохи
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    accumulate_steps = int(config.get('accumulate_steps', 1))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Перемещение данных на устройство
        images = batch['images'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_masks'].to(device)
        calories_per_100g_true = batch['calories_per_100g'].to(device)
        mass_features = batch['mass_features'].to(device)
        
        # Forward pass
        calories_pred_per_100g = model(
            images, text_input_ids, text_attention_mask, mass_features)
        
        # Новый лосс: MAE + 0.2*MSE для лучшей сходимости
        import torch.nn.functional as F
        mae_loss = F.l1_loss(calories_pred_per_100g, calories_per_100g_true)
        mse_loss = F.mse_loss(calories_pred_per_100g, calories_per_100g_true)
        loss = mae_loss + 0.2 * mse_loss
        
        # Backward pass
        (loss / accumulate_steps).backward()
        
        # Обновление и клиппинг по шагам аккумуляции
        if (batch_idx + 1) % accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()
        
        # Статистика - конверсия в общие калории для понятных метрик
        with torch.no_grad():
            # Конвертируем калории на 100г обратно в общие калории для метрики
            masses = batch['masses'].to(device)
            calories_pred_total = calories_pred_per_100g * (masses / 100.0)
            calories_true_total = calories_per_100g_true * (masses / 100.0)
            mae = F.l1_loss(calories_pred_total, calories_true_total).item()
        
        total_loss += loss.item()
        total_mae += mae
        
        # Логирование каждые N шагов
        log_every_n_steps = config.get('logging', {}).get('log_every_n_steps', 10)
        progress_info = {
            'Loss': f'{loss.item():.4f}',
            'MAE': f'{mae:.2f} ккал'
        }
        if batch_idx % log_every_n_steps == 0:
            progress_info['Step'] = batch_idx
            
        progress_bar.set_postfix(progress_info)
    
    return total_loss / num_batches, total_mae / num_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Валидирует модель на одной эпохе.
    
    Args:
        model: Модель для валидации
        val_loader: Загрузчик данных для валидации
        criterion: Функция потерь
        device: Устройство для вычислений
        
    Returns:
        Tuple с loss, MAE и MSE для эпохи
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            # Перемещение данных на устройство
            images = batch['images'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_masks'].to(device)
            calories_per_100g_true = batch['calories_per_100g'].to(device)
            masses = batch['masses'].to(device)
            mass_features = batch['mass_features'].to(device)
            
            # Forward pass
            calories_pred_per_100g = model(
                images, text_input_ids, text_attention_mask, mass_features)

            # Конвертация обратно в общие калории для валидации
            calories_pred_total = calories_pred_per_100g * (masses / 100.0)
            calories_true_total = calories_per_100g_true * (masses / 100.0)
            
            # Вычисление метрик
            mae_loss = F.l1_loss(calories_pred_per_100g,
                                 calories_per_100g_true)
            mse_loss = F.mse_loss(calories_pred_per_100g,
                                  calories_per_100g_true)
            loss = mae_loss + 0.2 * mse_loss

            # Метрики в общих калориях для понятности
            mae = F.l1_loss(calories_pred_total, calories_true_total).item()
            mse = F.mse_loss(calories_pred_total, calories_true_total).item()
            
            total_loss += loss.item()
            total_mae += mae
            total_mse += mse
    
    return total_loss / num_batches, total_mae / num_batches, total_mse / num_batches




def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    checkpoint_callback: Optional[callable] = None
) -> Dict[str, float]:
    """
    Выполняет полный цикл обучения модели с логированием и сохранением.
    
    Args:
        model: Модель для обучения
        optimizer: Оптимизатор
        scheduler: Планировщик learning rate
        criterion: Функция потерь
        train_loader: Загрузчик тренировочных данных
        val_loader: Загрузчик валидационных данных
        config: Конфигурация обучения
        device: Устройство для вычислений
        checkpoint_callback: Опциональная функция обратного вызова
        
    Returns:
        Словарь с лучшими метриками
    """
    print("=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 70)
    
    # Инициализация для отслеживания лучших результатов
    best_mae = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    # Параметры разморозки
    freeze_vision_epochs = int(config.get('freeze_vision_epochs', 0))
    freeze_text_epochs = int(config.get('freeze_text_epochs', 0))

    # Начинаем обучение
    start_time = time.time()
    # Переменные разморозки, определены выше
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        print(f"\nЭпоха {epoch + 1}/{config['epochs']}")
        print("-" * 50)
        
        # Разморозка энкодеров в указанные эпохи
        if epoch == freeze_vision_epochs and freeze_vision_epochs > 0:
            for p in model.vision_encoder.parameters():
                p.requires_grad = True
            print("Разморозили vision encoder")
        if epoch == freeze_text_epochs and freeze_text_epochs > 0:
            for p in model.text_encoder.parameters():
                p.requires_grad = True
            print("Разморозили text encoder")
        
        # ОБУЧЕНИЕ
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        
        # ВАЛИДАЦИЯ
        val_loss, val_mae, val_mse = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Обновляем learning rate
        scheduler.step()
        
        # Сохраняем метрики
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        # Вычисляем время эпохи
        epoch_time = time.time() - epoch_start
        
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f} ккал")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f} ккал, MSE: {val_mse:.2f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Время эпохи: {epoch_time:.1f} сек")
        
        # Вызываем callback функцию если предоставлена
        checkpoint_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_mse': val_mse,
            'best_mae': best_mae,
            'best_epoch': best_epoch,
            'is_best': False,
            'config': config
        }
        
        # Проверяем, стала ли это лучшей моделью
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            checkpoint_data['is_best'] = True
            
            print(f"✅ НОВАЯ ЛУЧШАЯ МОДЕЛЬ! MAE: {val_mae:.2f} ккал")
            
            # Проверяем достижение цели
            if val_mae < config['target_mae']:
                print(f"🎯 ЦЕЛЬ ДОСТИГНУТА! MAE < {config['target_mae']} ккал")
        
        # Вызываем callback для сохранения модели
        if checkpoint_callback:
            checkpoint_callback(model, optimizer, scheduler, checkpoint_data, train_losses, val_losses, train_maes, val_maes)
        
        # Early stopping
        if epoch - best_epoch > config['early_stopping_patience']:
            print(f"\n🛑 Early stopping на эпохе {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*70}")
    print(f"Лучшая MAE: {best_mae:.2f} ккал")
    print(f"Лучшая эпоха: {best_epoch + 1}")
    print(f"Общее время обучения: {total_time/3600:.2f} часов")
    
    return {
        'best_mae': best_mae,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'total_time': total_time
    }


def create_checkpoint_callback(config: Dict, device: torch.device):
    """
    Создает функцию обратного вызова для сохранения чекпоинтов.
    
    Args:
        config: Конфигурация модели
        device: Устройство для вычислений
        
    Returns:
        Функция callback для сохранения моделей
    """
    os.makedirs(config['output_dir'], exist_ok=True)
    
    def checkpoint_callback(model, optimizer, scheduler, data, train_losses, val_losses, train_maes, val_maes):
        epoch = data['epoch']
        epoch_num = epoch + 1
        
        # Проверяем настройки сохранения из конфига
        save_checkpoints = config['logging']['save_checkpoints']
        checkpoint_interval = config['logging']['checkpoint_interval']
        
        # Сохраняем чекпоинты эпох только если включено и по расписанию
        if save_checkpoints and checkpoint_interval >= 1:
            if epoch_num % checkpoint_interval == 0:
                epoch_model_path = os.path.join(config['output_dir'], f'model_epoch_{epoch_num:03d}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_mae': data['val_mae'],
                    'val_loss': data['val_loss'],
                    'train_mae': data['train_mae'],
                    'train_loss': data['train_loss'],
                    'config': config
                }, epoch_model_path)
                print(f"Модель эпохи {epoch_num} сохранена: {epoch_model_path}")
        
        # Сохраняем лучшую модель отдельно (всегда при улучшении MAE)
        if data['is_best']:
            best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_mae': data['val_mae'],
                'val_loss': data['val_loss'],
                'train_mae': data['train_mae'],
                'train_loss': data['train_loss'],
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_maes': train_maes,
                'val_maes': val_maes,
                'is_best_model': True
            }, best_model_path)
            print(f"ЛУЧШАЯ модель обновлена: {best_model_path} (MAE: {data['val_mae']:.2f})")
    
    return checkpoint_callback


def train(
    config_path: str,
    output_dir: str = "models"
) -> Dict[str, float]:
    """
    Главная функция обучения модели.
    
    Args:
        config_path: Путь к файлу конфигурации
        output_dir: Директория для сохранения модели
        
    Returns:
        Словарь с лучшими метриками
    """
    # Загрузка конфигурации
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Установка seed для воспроизводимости
    seed_everything(config['seed'])
    
    # Создание директории для модели
    os.makedirs(output_dir, exist_ok=True)
    
    # Определение устройства из конфига
    device_setting = config.get('device', 'auto')
    if device_setting == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_setting == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_setting == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(device_setting)
        
    print(f"Используется устройство: {device}")
    
    # Загрузка данных из путей в конфиге
    print("Загрузка данных...")
    import pandas as pd
    
    data_paths = config.get('data_paths', {
        'dish_csv': 'data/dish.csv',
        'ingredients_csv': 'data/ingredients.csv',
        'images_dir': 'data/images'
    })
    
    dish_df = pd.read_csv(data_paths['dish_csv'])
    ingredients_df = pd.read_csv(data_paths['ingredients_csv'])
    
    # Создание загрузчиков данных с передачей конфига для аугментаций
    train_loader, val_loader = create_data_loaders(
        dish_df=dish_df,
        ingredients_df=ingredients_df,
        image_dir=data_paths['images_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        config=config  # Передаем конфиг для аугментаций и image_size
    )
    
    # Создание модели
    print("Инициализация модели...")
    model = CaloriePredictor(
        vision_model_name=config['vision_model'],
        text_model_name=config['text_model'],
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Этапная разморозка энкодеров (опционально из конфига)
    freeze_vision_epochs = int(config.get('freeze_vision_epochs', 0))
    freeze_text_epochs = int(config.get('freeze_text_epochs', 0))
    if freeze_vision_epochs > 0:
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
    if freeze_text_epochs > 0:
        for p in model.text_encoder.parameters():
            p.requires_grad = False
    
    # Параметры разморозки
    freeze_vision_epochs = int(config.get('freeze_vision_epochs', 0))
    freeze_text_epochs = int(config.get('freeze_text_epochs', 0))
    if freeze_vision_epochs > 0:
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        print(f"Заморозили vision encoder на {freeze_vision_epochs} эпох")
    if freeze_text_epochs > 0:
        for p in model.text_encoder.parameters():
            p.requires_grad = False
        print(f"Заморозили text encoder на {freeze_text_epochs} эпох")

    # Создание оптимизатора и планировщика
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Функция потерь
    criterion = nn.SmoothL1Loss(reduction='mean')
    
    # Обучение
    print("Начало обучения...")
    best_mae = float('inf')
    best_metrics = {}
    
    for epoch in range(config['epochs']):
        print(f"\nЭпоха {epoch + 1}/{config['epochs']}")
        
        # Обучение
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        
        # Валидация
        val_loss, val_mae, val_mse = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Обновление learning rate
        scheduler.step()
        
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, MSE: {val_mse:.2f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Сохранение лучшей модели
        if val_mae < best_mae:
            best_mae = val_mae
            best_metrics = {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'epoch': epoch
            }
            
            # Сохранение модели
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_mae': best_mae,
                'config': config
            }, model_path)
            
            print(f"Новая лучшая модель сохранена! MAE: {val_mae:.2f}")
        
        # Early stopping
        if epoch - best_metrics['epoch'] > config['early_stopping_patience']:
            print(f"Early stopping на эпохе {epoch + 1}")
            break
    
    print(f"\nОбучение завершено!")
    print(f"Лучшая MAE: {best_mae:.2f}")
    print(f"Лучшая эпоха: {best_metrics['epoch'] + 1}")
    
    return best_metrics


if __name__ == '__main__':
    # Пример запуска
    import sys
    from pathlib import Path
    
    # Добавляем корневую директорию в путь
    root_path = Path(__file__).parent.parent
    sys.path.append(str(root_path))
    
    # Загружаем и запускаем обучение
    metrics = train(
        config_path=str(root_path / "configs/config.yaml"),
        output_dir="models"
    )
    print("Финальные метрики:", metrics)
