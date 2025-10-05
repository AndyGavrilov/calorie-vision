"""Скрипт для запуска обучения модели предсказания калорийности блюд.

Использование:
    python train.py --config configs/config.yaml --output_dir models
    python train.py  # использует параметры по умолчанию
"""

import argparse
import sys
from pathlib import Path

# Добавляем папку scripts в путь для импорта модулей
root_path = Path(__file__).parent
sys.path.append(str(root_path / "scripts"))

from scripts.utils import train


def main():
    """Основная функция для запуска обучения."""
    parser = argparse.ArgumentParser(
        description="Обучение модели предсказания калорийности блюд"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Путь к конфигурационному файлу"
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default="models",
        help="Директория для сохранения обученной модели"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости"
    )
    
    args = parser.parse_args()
    
    # Проверяем существование конфигурационного файла
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Ошибка: Конфигурационный файл {config_path} не найден!")
        sys.exit(1)
    
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ КАЛОРИЙНОСТИ БЛЮД")
    print("=" * 60)
    print(f"Конфигурация: {args.config}")
    print(f"Выходная директория: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    try:
        # Запуск обучения
        metrics = train(
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("=" * 60)
        print(f"Лучшая MAE на валидации: {metrics['val_mae']:.2f} ккал")
        print(f"Лучшая Loss на валидации: {metrics['val_loss']:.4f}")
        print(f"Лучшая MSE на валидации: {metrics['val_mse']:.2f}")
        print(f"Лучшая эпоха: {metrics['epoch'] + 1}")
        
        # Проверка достижения цели
        target_mae = 50.0
        if metrics['val_mae'] < target_mae:
            print(f"ЦЕЛЬ ДОСТИГНУТА! MAE < {target_mae} ккал")
        else:
            print(f"Цель НЕ достигнута. MAE >= {target_mae} ккал")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\nОШИБКА ОБУЧЕНИЯ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
