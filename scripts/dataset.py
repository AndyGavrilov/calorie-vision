"""Датасет для задачи предсказания калорийности блюд"""

import os
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import timm
import albumentations as A
import albumentations.pytorch as APT
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


class CalorieDataset(Dataset):
    """Мультимодальный датасет для предсказания калорийности блюд.
    
    Данный класс обрабатывает как изображения блюд, так и список ингредиентов
    для предсказания калорийности.
    """
    
    def __init__(
        self, 
        dish_df: pd.DataFrame,
        ingredients_df: pd.DataFrame,
        image_dir: str,
        transforms: Optional[A.Compose] = None,
        mode: str = "train",
        tokenizer_name: Optional[str] = None,
        text_max_length: int = 256
    ):
        """
        Args:
            dish_df: DataFrame с данными о блюдах
            ingredients_df: DataFrame с названиями ингредиентов
            image_dir: Путь к директории с изображениями
            transforms: Аугментации для изображений
            mode: Режим работы ("train", "val", "test")
        """
        self.dish_df = dish_df
        self.ingredients_df = ingredients_df
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode
        self.text_max_length = text_max_length
        
        # Фильтруем данные по режиму
        self.filtered_df = dish_df[dish_df['split'] == mode].reset_index(drop=True)
        
        # Создаем маппинг ID -> название ингредиента
        self.ingredient_mapping = dict(zip(
            ingredients_df['id'].astype(int),
            ingredients_df['ingr']
        ))
        
        # Токенизатор для обработки ингредиентов (берем из конфига, если передан)
        tokenizer_model_name = tokenizer_name or 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        
        print(f"Загружено {len(self.filtered_df)} образцов для режима {mode}")
    
    def __len__(self):
        return len(self.filtered_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Возвращает образец данных.
        
        Args:
            idx: Индекс образца
            
        Returns:
            Словарь с данными для модели:
            - image: тензор изображения
            - text_input_ids: токенизированные ингредиенты
            - text_attention_mask: маска внимания
            - calories: целевая переменная (калорийность)
            - mass: масса блюда
        """
        row = self.filtered_df.iloc[idx]
        
        # Загрузка изображения
        image_path = os.path.join(self.image_dir, row['dish_id'], 'rgb.png')
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            # Создаем пустое изображение в случае ошибки
            image_array = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Применение аугментаций
        if self.transforms:
            augmented = self.transforms(image=image_array)
            image = augmented['image']
        else:
            image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        # Обработка ингредиентов
        ingredients_text = self.convert_ingredients_to_text(row['ingredients'])
        
        # Токенизация текста ингредиентов
        text_encoded = self.tokenizer(
            ingredients_text,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Новый таргет: калории на 100г (более стабильная метрика)
        calories_per_100g = row['total_calories'] / (row['total_mass'] / 100.0)

        # Устойчивое лог-нормализация массы для признака
        mass_feature = np.log1p(row['total_mass']).astype(np.float32)

        return {
            'image': image,
            'text_input_ids': text_encoded['input_ids'].squeeze(0),
            'text_attention_mask': text_encoded['attention_mask'].squeeze(0),
            # новый таргет
            'calories_per_100g': torch.tensor(calories_per_100g, dtype=torch.float32),
            # для обратного перевода
            'mass': torch.tensor(row['total_mass'], dtype=torch.float32),
            # признак для модели
            'mass_feature': torch.tensor(mass_feature, dtype=torch.float32),
            'dish_id': row['dish_id']
        }
    
    def convert_ingredients_to_text(self, ingredients_str: str) -> str:
        """
        Конвертирует строку с ID ингредиентов в текстовое представление.
        
        Args:
            ingredients_str: Строка вида "ingr_0000000508;ingr_0000000122"
            
        Returns:
            Текстовое представление ингредиентов
        """
        ingredient_ids = ingredients_str.split(';')
        ingredients = []
        
        for ingr_id in ingredient_ids:
            # Извлекаем номер из ID
            id_str = ingr_id.replace('ingr_', '')
            try:
                key = int(id_str)
            except:
                key = None
            if key in self.ingredient_mapping:
                ingredients.append(self.ingredient_mapping[key])
            else:
                ingredients.append("unknown ingredient")
        
        return ", ".join(ingredients)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Функция для группировки батча образцов.
    
    Args:
        batch: Список словарей с данными образцов
        
    Returns:
        Батч тензоров для модели
    """
    images = torch.stack([item['image'] for item in batch])
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_masks = torch.stack([item['text_attention_mask'] for item in batch])
    calories_per_100g = torch.stack(
        [item['calories_per_100g'] for item in batch])
    masses = torch.stack([item['mass'] for item in batch])
    mass_features = torch.stack([item['mass_feature'] for item in batch])
    dish_ids = [item['dish_id'] for item in batch]
    
    return {
        'images': images,
        'text_input_ids': text_input_ids,
        'text_attention_masks': text_attention_masks,
        'calories_per_100g': calories_per_100g,
        'masses': masses,
        'mass_features': mass_features,
        'dish_ids': dish_ids
    }


def get_transforms(mode: str = "train", image_size: int = 224, config: dict = None) -> A.Compose:
    """
    Возвращает набор аугментаций для изображений на основе конфига или дефолтных значений.
    
    Args:
        mode: Режим работы ("train", "val", "test")
        image_size: Размер изображения после resize
        config: Словарь с настройками аугментаций
        
    Returns:
        Compose объект Albumentations
    """
    # Автоматическое определение оптимального разрешения из модели
    try:
        cfg = timm.get_pretrained_cfg('efficientnet_b3')
        optimal_size = max(cfg.input_size[1], cfg.input_size[2])
        print(f"Оптимальное разрешение для EfficientNet-B3: {optimal_size}")
    except:
        optimal_size = image_size
    
    # Дефолтные значения если конфиг не передан
    if config is None:
        augmentation_config = {
            'image_size': image_size,
            'horizontal_flip_prob': 0.5,
            'color_jitter_prob': 0.9,
            'color_jitter_brightness': 0.4,
            'color_jitter_contrast': 0.4,
            'color_jitter_saturation': 0.4,
            'color_jitter_hue': 0.15,
            'rotation_range': [-15, 15],
            'elastic_transform_prob': 0.3,
            'affine_prob': 0.8,
            'affine_scale': (0.8, 1.2),
            'affine_translate': (-0.1, 0.1),
            'coarse_dropout_prob': 0.5,
            'coarse_dropout_holes_range': (2, 8),
            'transform_seed': 42
        }
        final_image_size = image_size
    else:
        augmentation_config = config.get('augmentation', {})
        final_image_size = augmentation_config.get('image_size', config.get('image_size', image_size))
    
    if mode == "train":
        transforms = A.Compose([
            # Resize: сначала делаем изображение большего размерами, потом случайный кроп
            A.SmallestMaxSize(max_size=final_image_size + 32),
            A.RandomCrop(height=final_image_size, width=final_image_size),
            
            # Горизонтальное отражение
            A.HorizontalFlip(p=augmentation_config.get('horizontal_flip_prob', 0.5)),
            
            # Геометрические трансформации
            A.Affine(
                scale=augmentation_config.get('affine_scale', (0.8, 1.2)),
                rotate=augmentation_config.get('rotation_range', [-15, 15]),
                translate_percent=augmentation_config.get('affine_translate', (-0.1, 0.1)),
                shear=(-10, 10),
                fill=0,
                p=augmentation_config.get('affine_prob', 0.8)
            ),
            
            # Цветовые трансформации
            A.ColorJitter(
                brightness=augmentation_config.get('color_jitter_brightness', 0.4),
                contrast=augmentation_config.get('color_jitter_contrast', 0.4),
                saturation=augmentation_config.get('color_jitter_saturation', 0.4),
                hue=augmentation_config.get('color_jitter_hue', 0.15),
                p=augmentation_config.get('color_jitter_prob', 0.9)
            ),
            
            # CoarseDropout с переменным количеством дыр
            A.CoarseDropout(
                num_holes_range=augmentation_config.get('coarse_dropout_holes_range', (2, 8)),
                hole_height_range=augmentation_config.get('coarse_dropout_height_range', 
                    (int(0.07 * final_image_size), int(0.15 * final_image_size))),
                hole_width_range=augmentation_config.get('coarse_dropout_width_range', 
                    (int(0.1 * final_image_size), int(0.15 * final_image_size))),
                fill=0,
                p=augmentation_config.get('coarse_dropout_prob', 0.5)
            ),
            
            # Эластичные искажения
            A.ElasticTransform(
                p=augmentation_config.get('elastic_transform_prob', 0.3)
            ),
            
            # Нормализация для предобученных моделей
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]   # ImageNet std
            ),
            APT.ToTensorV2()  # Преобразование в тензор
        ], seed=augmentation_config.get('transform_seed', 42)) 
    else:
        # Валидация/тест: только resize и normalize, без аугментаций
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=final_image_size),
            A.CenterCrop(height=final_image_size, width=final_image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            APT.ToTensorV2()
        ], seed=augmentation_config.get('transform_seed', 42))
    
    return transforms


def create_data_loaders(
    dish_df: pd.DataFrame,
    ingredients_df: pd.DataFrame,
    image_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    config: dict = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Создает DataLoader для обучения и валидации.
    
    Args:
        dish_df: DataFrame с данными о блюдах
        ingredients_df: DataFrame с названиями ингредиентов
        image_dir: Путь к директории с изображениями
        batch_size: Размер батча
        num_workers: Количество процессов для загрузки данных
        config: Словарь с конфигурацией (для передачи image_size и аугментаций)
        
    Returns:
        Tuple с train и validation DataLoader
    """
    # Получаем image_size из конфига или используем дефолт
    image_size = config.get('image_size', 224) if config else 224
    
    # Создание датасетов
    train_dataset = CalorieDataset(
        dish_df=dish_df,
        ingredients_df=ingredients_df,
        image_dir=image_dir,
        transforms=get_transforms(mode="train", image_size=image_size, config=config),
        mode="train",
        tokenizer_name=config.get('text_model') if config else None,
        text_max_length=config.get('text_max_length', 256) if config else 256
    )
    
    val_dataset = CalorieDataset(
        dish_df=dish_df,
        ingredients_df=ingredients_df,
        image_dir=image_dir,
        transforms=get_transforms(mode="val", image_size=image_size, config=config),
        mode="test",
        tokenizer_name=config.get('text_model') if config else None,
        text_max_length=config.get('text_max_length', 256) if config else 256
    )
    
    # Создание DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader
