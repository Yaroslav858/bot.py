import os
import logging
import time
import sys
import re
import sqlite3
import requests
import asyncio
import json
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pickle
import csv
import openpyxl
import pandas as pd
#from File_Manager import FileManager
from openpyxl import load_workbook
from telegram import Update
import yaml
import xml.etree.ElementTree as ET
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, ConversationHandler, CallbackQueryHandler, 
    MessageHandler, filters, ContextTypes
)
import subprocess
import shutil
import psutil
from pathlib import Path
from datetime import timedelta
from enum import Enum





# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = "6895869913:AAFNEmshnKg2Dd9-GGd6q5z1ygX3gAaUqvI"
ADMIN_IDS = [6424735984]
SUPPORT_GROUP_ID = "@moto_angel1"

# Hugging Face настройки (резервный вариант)
HUGGINGFACE_API_KEY = "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"
HUGGING_FACE_MODELS = [
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-small", 
    "microsoft/DialoGPT-large",
    "facebook/blenderbot-400M-distill"
]



class FileManager:
    """Простой менеджер файлов для загрузки"""
    
    def __init__(self):
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def upload_file(self, file_id, subject_id, file_type, file_name, user_id):
        """Загрузка файла с базовой логикой"""
        try:
            # Создаем директории по типам
            type_dir = "lectures" if file_type == "lecture" else "practices" if file_type == "practice" else "materials"
            base_dir = os.path.join(self.upload_dir, type_dir)
            os.makedirs(base_dir, exist_ok=True)
            
            # Сохраняем файл
            file_path = os.path.join(base_dir, file_name)
            
            # В реальной реализации здесь должна быть логика скачивания файла из Telegram
            # и сохранения на диск. Для примера просто создаем пустой файл.
            with open(file_path, 'w') as f:
                f.write(f"File: {file_name}\nSubject: {subject_id}\nType: {file_type}")
            
            logger.info(f"Файл сохранен: {file_path}")
            
            return {
                'success': True,
                'file_path': file_path,
                'message': f"✅ Файл '{file_name}' успешно загружен"
            }
            
        except Exception as e:
            logger.error(f"Ошибка загрузки файла: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_file_path(self, file_type, file_name):
        """Получить путь к файлу"""
        type_dir = "lectures" if file_type == "lecture" else "practices" if file_type == "practice" else "materials"
        return os.path.join(self.upload_dir, type_dir, file_name)


# =============================================================================
# СИСТЕМА УДАЛЕННОГО УПРАВЛЕНИЯ КОДОМ (BotCodeManager)
# =============================================================================
class BotCodeManager:
    """Система удаленного управления кодом бота"""
    def __init__(self, bot_instance=None):
        self.bot_instance = bot_instance
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Получить статус системы"""
        try:
            # Информация о системе
            system_info = {
                "platform": sys.platform,
                "python_version": sys.version,
                "bot_uptime": self._get_bot_uptime(),
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if sys.platform != 'win32' else psutil.disk_usage('C:').percent,
                "active_processes": len(psutil.pids())
            }
            
            # Информация о боте
            bot_info = {
                "database_size": self._get_database_size(),
                "log_files_count": len(self._get_log_files()),
                "training_datasets_count": len(self._get_training_datasets()),
                "ai_model_status": "✅ Активен" if os.path.exists("self_learning_model.pth") else "❌ Не обучен"
            }
            
            return {
                "system": system_info,
                "bot": bot_info,
                "timestamp": datetime.now().isoformat(),
                "status": "✅ Система работает нормально"
            }
        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return {"error": str(e), "status": "❌ Ошибка системы"}
    
    def _get_bot_uptime(self) -> str:
        """Получить время работы бота"""
        try:
            if hasattr(self.bot_instance, 'start_time'):
                uptime = datetime.now() - self.bot_instance.start_time
                return str(uptime).split('.')[0]
        except:
            pass
        return "Неизвестно"
    
    def _get_database_size(self) -> str:
        """Получить размер базы данных"""
        try:
            if os.path.exists("bot_database.db"):
                size = os.path.getsize("bot_database.db")
                return f"{size / 1024 / 1024:.2f} MB"
        except:
            pass
        return "Неизвестно"
    
    def _get_log_files(self) -> List[str]:
        """Получить список лог-файлов"""
        try:
            return [f for f in os.listdir('.') if f.endswith('.log')]
        except:
            return []
    
    def _get_training_datasets(self) -> List[str]:
        """Получить список датасетов"""
        try:
            datasets_dir = "training_datasets"
            if os.path.exists(datasets_dir):
                return os.listdir(datasets_dir)
        except:
            pass
        return []
    
    async def cleanup_temp_files(self) -> Tuple[bool, str]:
        """Асинхронная версия очистки временных файлов"""
        try:
            temp_dirs = ['temp_update', 'temp_download', '__pycache__']
            cleaned = []
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        if os.path.isdir(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            cleaned.append(temp_dir)
                        else:
                            os.remove(temp_dir)
                            cleaned.append(temp_dir)
                    except Exception as e:
                        logger.warning(f"Не удалось очистить {temp_dir}: {e}")
            
            # Очищаем файлы .pyc
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pyc'):
                        try:
                            os.remove(os.path.join(root, file))
                            cleaned.append(file)
                        except:
                            pass
            
            if cleaned:
                return True, f"✅ Очищены временные файлы: {', '.join(cleaned)}"
            else:
                return True, "✅ Временные файлы не найдены"
        except Exception as e:
            logger.error(f"Ошибка очистки временных файлов: {e}")
            return False, f"❌ Ошибка очистки: {str(e)}"
        
    def create_backup_sync(self) -> Tuple[bool, str]:
        """Синхронная версия создания бэкапа"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            os.makedirs(backup_path, exist_ok=True)
            
            # Копируем важные файлы
            important_files = [
                'bot_database.db',
                'self_learning_model.pth',
                'dataset_vectorizer.pkl',
                'model_info.json'
            ]
            
            for file in important_files:
                if os.path.exists(file):
                    shutil.copy2(file, os.path.join(backup_path, file))
            
            # Копируем директории
            important_dirs = ['training_datasets', 'schedules', 'useful_info']
            for dir_name in important_dirs:
                if os.path.exists(dir_name):
                    shutil.copytree(
                        dir_name, 
                        os.path.join(backup_path, dir_name),
                        dirs_exist_ok=True
                    )
            
            # Создаем файл с информацией о бэкапе
            backup_info = {
                "timestamp": datetime.now().isoformat(),
                "files": important_files + important_dirs,
                "total_size": self._get_dir_size(backup_path)
            }
            
            with open(os.path.join(backup_path, "backup_info.json"), 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            return True, f"✅ Бэкап создан: {backup_name}"
            
        except Exception as e:
            logger.error(f"Ошибка создания бэкапа: {e}")
            return False, f"❌ Ошибка создания бэкапа: {str(e)}"

    

    def _get_dir_size(self, path: str) -> str:
        """Получить размер директории"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return f"{total / 1024 / 1024:.2f} MB"
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """Получить список бэкапов"""
        backups = []
        try:
            for item in os.listdir(self.backup_dir):
                backup_path = os.path.join(self.backup_dir, item)
                if os.path.isdir(backup_path):
                    info_file = os.path.join(backup_path, "backup_info.json")
                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        backups.append({
                            "name": item,
                            "timestamp": info.get("timestamp", ""),
                            "size": info.get("total_size", "0 MB")
                        })
        except Exception as e:
            logger.error(f"Ошибка получения списка бэкапов: {e}")
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

    # Добавляем остальные методы, которые были в оригинальном классе
    async def create_backup(self) -> Tuple[bool, str]:
        """Асинхронная версия создания бэкапа"""
        return await asyncio.get_event_loop().run_in_executor(
        None, self.create_backup_sync
    )

    def view_file(self, file_path: str) -> Tuple[bool, str, str]:
        """Просмотреть содержимое файла"""
        try:
            if not os.path.exists(file_path):
                return False, "", f"❌ Файл не найден: {file_path}"
            
            # Проверяем размер файла
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024:  # 100KB лимит
                return False, "", f"❌ Файл слишком большой ({file_size} байт). Максимум 100KB."
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return True, content, f"✅ Файл прочитан: {file_path}"
            
        except Exception as e:
            logger.error(f"Ошибка чтения файла: {e}")
            return False, "", f"❌ Ошибка чтения файла: {str(e)}"

    async def edit_file(self, file_path: str, new_content: str) -> Tuple[bool, str]:
        """Редактировать файл"""
        try:
            if not os.path.exists(file_path):
                return False, f"❌ Файл не найден: {file_path}"
            
            # Создаем бэкап файла
            backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)
            
            # Записываем новый контент
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True, f"✅ Файл успешно обновлен. Бэкап: {backup_path}"
            
        except Exception as e:
            logger.error(f"Ошибка редактирования файла: {e}")
            return False, f"❌ Ошибка редактирования: {str(e)}"

    async def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """Выполнить команду на сервере"""
        try:
            # Безопасность: ограничиваем опасные команды
            dangerous_commands = ['rm -rf', 'format', 'dd', 'mkfs', 'chmod 777']
            if any(cmd in command for cmd in dangerous_commands):
                return False, "", "❌ Опасная команда запрещена"
            
            # Специальная команда для очистки временных файлов
            if command.strip() == "cleanup":
                success, message = await self.cleanup_temp_files()
                return success, message, "Команда очистки выполнена"
            
            # Выполняем команду
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            output = result.stdout if result.stdout else result.stderr
            success = result.returncode == 0
            
            return success, output, f"Код возврата: {result.returncode}"
            
        except subprocess.TimeoutExpired:
            return False, "", "❌ Команда превысила лимит времени (30 секунд)"
        except Exception as e:
            logger.error(f"Ошибка выполнения команды: {e}")
            return False, "", f"❌ Ошибка выполнения: {str(e)}"

    def restart_bot(self) -> Tuple[bool, str]:
        """Перезапустить бота"""
        try:
            # Заменить await self.bot_instance.shutdown() на:
            if self.bot_instance:
                # Если есть метод shutdown, вызвать его
                if hasattr(self.bot_instance, 'shutdown'):
                    self.bot_instance.shutdown()
            
            subprocess.Popen([sys.executable, __file__])
            sys.exit(0)
            
            return True, "✅ Бот перезапускается..."
        except Exception as e:
            logger.error(f"Ошибка перезапуска бота: {e}")
            return False, f"❌ Ошибка перезапуска: {str(e)}"

# =============================================================================
# КЛАСС SIMPLETEXTVECTORIZER
# =============================================================================

class SimpleTextVectorizer:
    """Простой векторizer текстов без scikit-learn"""
    
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary = {}
        self.stop_words = self.get_stop_words()
        
    def get_stop_words(self):
        """Стоп-слова для русского и английского языков"""
        russian_stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 
            'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 
            'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 
            'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 
            'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 
            'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 
            'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 
            'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 
            'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'неё', 'сейчас', 'были', 
            'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 
            'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 
            'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 
            'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 
            'конечно', 'всю', 'между'
        }
        
        english_stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
            'wouldn', "wouldn't"
        }
        
        return russian_stop_words.union(english_stop_words)
    
    def build_vocabulary(self, texts):
        """Построение словаря из текстов"""
        if not texts:
            logger.warning("Нет текстов для построения словаря")
            return
            
        word_counts = Counter()
        
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            words = self.preprocess_text(text).split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            word_counts.update(words)
        
        if not word_counts:
            logger.warning("Не удалось извлечь слова из текстов")
            return
        
        # Берем самые частые слова
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(most_common)}
        
        logger.info(f"Построен словарь из {len(self.vocabulary)} слов")
    
    def preprocess_text(self, text: str) -> str:
        """Предобработка текста"""
        if not text:
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower().strip()
        
        # Удаляем специальные символы, но сохраняем буквы и пробелы
        text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', ' ', text)
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def transform(self, texts):
        """Преобразование текстов в векторы"""
        if not self.vocabulary:
            raise ValueError("Словарь не построен. Сначала вызовите build_vocabulary")
        
        vectors = []
        for text in texts:
            if not text:
                vectors.append(np.zeros(len(self.vocabulary)))
                continue
                
            vector = np.zeros(len(self.vocabulary))
            words = self.preprocess_text(text).split()
            words = [word for word in words if word in self.vocabulary]
            
            for word in words:
                vector[self.vocabulary[word]] += 1
            
            # Нормализуем вектор (TF)
            if np.sum(vector) > 0:
                vector = vector / np.sum(vector)
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts):
        """Построение словаря и преобразование текстов"""
        self.build_vocabulary(texts)
        return self.transform(texts)
    
    def save(self, filename):
        """Сохранение векторizer'а"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'vocabulary': self.vocabulary,
                    'max_features': self.max_features,
                    'stop_words': self.stop_words
                }, f)
            logger.info(f"Векторизатор сохранен: {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения векторзатора: {e}")
    
    def load(self, filename):
        """Загрузка векторizer'а"""
        try:
            if not os.path.exists(filename):
                return False
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.vocabulary = data['vocabulary']
                self.max_features = data['max_features']
                self.stop_words = data['stop_words']
            logger.info(f"Векторизатор загружен: {filename}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки векторзатора: {e}")
            return False

# =============================================================================
# ДОБАВЛЕННЫЕ ФУНКЦИИ ДЛЯ УПРАВЛЕНИЯ ДАТАСЕТАМИ
# =============================================================================

class DatasetManager:
    """Менеджер для управления датасетами"""
    
    def __init__(self):
        self.datasets_dir = "training_datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Получить список всех датасетов с информацией"""
        try:
            files = os.listdir(self.datasets_dir)
            datasets = []
            
            for file in files:
                filepath = os.path.join(self.datasets_dir, file)
                if os.path.isfile(filepath):
                    file_size = os.path.getsize(filepath)
                    file_extension = os.path.splitext(file)[1].lower()
                    
                    datasets.append({
                        'filename': file,
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'extension': file_extension,
                        'created_time': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            return sorted(datasets, key=lambda x: x['filename'])
        except Exception as e:
            logger.error(f"Ошибка получения списка датасетов: {e}")
            return []
    
    def delete_dataset(self, filename: str) -> bool:
        """Удалить датасет"""
        # Добавляем проверку прав доступа
        
        try:
            filepath = os.path.join(self.datasets_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Датасет удален: {filename}")
                return True
            else:
                logger.warning(f"Файл датасета не найден: {filename}")
                return False
        except Exception as e:
            logger.error(f"Ошибка удаления датасета {filename}: {e}")
            return False
    
    def get_dataset_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Получить информацию о датасете"""
        try:
            filepath = os.path.join(self.datasets_dir, filename)
            if not os.path.exists(filepath):
                return None
            
            file_size = os.path.getsize(filepath)
            file_extension = os.path.splitext(filename)[1].lower()
            created_time = datetime.fromtimestamp(os.path.getctime(filepath))
            
            return {
                'filename': filename,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'size_gb': round(file_size / (1024 * 1024 * 1024), 3),
                'extension': file_extension,
                'created_time': created_time.strftime('%Y-%m-%d %H:%M:%S'),
                'days_old': (datetime.now() - created_time).days
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о датасете {filename}: {e}")
            return None

# =============================================================================
# ИСПРАВЛЕННЫЙ КЛАСС ДЛЯ ЗАГРУЗКИ С GITHUB
# =============================================================================

class AdvancedDatasetLoader:
    """Расширенный загрузчик датасетов с поддержкой множества форматов"""
    
    def __init__(self):
        self.datasets_dir = "training_datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)
        self.supported_formats = {
            'json': ['.json', '.jsonl', '.ndjson'],
            'csv': ['.csv', '.tsv', '.txt'],
            'excel': ['.xlsx', '.xls', '.xlsm'],
            'text': ['.txt', '.text', '.md'],
            'yaml': ['.yaml', '.yml'],
            'xml': ['.xml'],
            'parquet': ['.parquet'],
            'feather': ['.feather'],
            'pickle': ['.pkl', '.pickle']
        }
    
    def get_all_supported_extensions(self) -> List[str]:
        """Получить все поддерживаемые расширения"""
        all_extensions = []
        for extensions in self.supported_formats.values():
            all_extensions.extend(extensions)
        return all_extensions
    
    def detect_format(self, filename: str) -> Optional[str]:
        """Определить формат файла по расширению"""
        ext = os.path.splitext(filename)[1].lower()
        for format_type, extensions in self.supported_formats.items():
            if ext in extensions:
                return format_type
        return None
    
    async def download_from_github(self, url: str) -> Optional[str]:
        """Скачать датасет с GitHub (использует улучшенный GitHubDatasetLoader)"""
        github_loader = GitHubDatasetLoader()
        return await github_loader.download_from_github(url)
    
    def load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка датасета из файла с улучшенной обработкой ошибок"""
        try:
            filepath = os.path.join(self.datasets_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"Файл датасета не найден: {filepath}")
                return np.array([]), np.array([])
            
            file_format = self.detect_format(filename)
            
            # Для JSON файлов сначала делаем диагностику если есть проблемы
            if file_format == 'json':
                # Проверяем размер файла
                file_size = os.path.getsize(filepath)
                if file_size == 0:
                    logger.error(f"JSON файл пустой: {filename}")
                    return np.array([]), np.array([])
                
                # Если файл очень большой, используем потоковую обработку
                if file_size > 10 * 1024 * 1024:  # 10MB
                    logger.info(f"Большой JSON файл ({file_size} байт), используем потоковую обработку")
                    return self._load_large_json(filepath)
            
            # Стандартная загрузка для других форматов
            if file_format == 'json':
                return self._load_json_dataset(filepath)
            elif file_format == 'csv':
                return self._load_csv_dataset(filepath)
            elif file_format == 'excel':
                return self._load_excel_dataset(filepath)
            elif file_format == 'text':
                return self._load_text_dataset(filepath)
            elif file_format == 'yaml':
                return self._load_yaml_dataset(filepath)
            elif file_format == 'xml':
                return self._load_xml_dataset(filepath)
            elif file_format == 'parquet':
                return self._load_parquet_dataset(filepath)
            elif file_format == 'feather':
                return self._load_feather_dataset(filepath)
            elif file_format == 'pickle':
                return self._load_pickle_dataset(filepath)
            else:
                logger.error(f"Неподдерживаемый формат файла: {filename}")
                return np.array([]), np.array([])
            
        except Exception as e:
            logger.error(f"Общая ошибка загрузки датасета {filename}: {e}")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
        
        # Пробуем диагностировать проблему для JSON файлов
        if filename.lower().endswith(('.json', '.jsonl')):
            filepath = os.path.join(self.datasets_dir, filename)
            diagnosis = self.diagnose_json_file(filepath)
            logger.error(f"Диагностика JSON файла: {diagnosis}")
        
        return np.array([]), np.array([])
    
    def _load_json_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка JSON датасета с улучшенной обработкой ошибок"""
        try:
            texts = []
            labels = []
            
            # Сначала пробуем определить формат файла
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)  # Возвращаемся в начало файла
                
                # Пробуем разные форматы JSON
                if first_line.startswith('['):
                    # Формат JSON массива
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                self._extract_from_item(item, texts, labels)
                        else:
                            # Если это не список, обрабатываем как обычный JSON
                            self._extract_from_item(data, texts, labels)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ошибка парсинга JSON массива: {e}")
                        # Пробуем как JSONL
                        return self._load_jsonl_dataset(filepath)
                        
                elif first_line.startswith('{'):
                    # Одиночный JSON объект
                    try:
                        data = json.load(f)
                        self._extract_from_item(data, texts, labels)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ошибка парсинга JSON объекта: {e}")
                        # Пробуем как JSONL
                        return self._load_jsonl_dataset(filepath)
                else:
                    # Вероятно JSONL или другой формат
                    return self._load_jsonl_dataset(filepath)
            
            logger.info(f"Успешно загружено JSON: {len(texts)} текстов, {len(set(labels))} меток")
            return self._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки JSON датасета {filepath}: {e}")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
            return np.array([]), np.array([])
    
    def _load_jsonl_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка JSONL/NDJSON датасета с улучшенной обработкой ошибок"""
        try:
            texts = []
            labels = []
            line_count = 0
            error_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    line_count += 1
                    
                    try:
                        # Пробуем разные кодировки если UTF-8 не работает
                        try:
                            item = json.loads(line)
                        except UnicodeDecodeError:
                            # Пробуем другие кодировки
                            for encoding in ['utf-8-sig', 'latin-1', 'cp1251']:
                                try:
                                    line_encoded = line.encode('utf-8').decode(encoding)
                                    item = json.loads(line_encoded)
                                    break
                                except:
                                    continue
                            else:
                                raise json.JSONDecodeError("Не удалось декодировать строку", line, 0)
                        
                        self._extract_from_item(item, texts, labels)
                        
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        error_count += 1
                        if error_count <= 5:  # Логируем только первые 5 ошибок
                            logger.warning(f"Ошибка парсинга JSON строки {line_num}: {e}")
                        continue
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            logger.warning(f"Другая ошибка в строке {line_num}: {e}")
                        continue
            
            if error_count > 0:
                logger.warning(f"Всего ошибок парсинга: {error_count} из {line_count} строк")
            
            if not texts:
                logger.error("Не удалось извлечь данные из JSONL файла")
                return np.array([]), np.array([])
            
            logger.info(f"Успешно загружено JSONL: {len(texts)} текстов, {len(set(labels))} меток")
            return self._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки JSONL датасета {filepath}: {e}")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
            return np.array([]), np.array([])
    
    def diagnose_json_file(self, filepath: str) -> Dict[str, Any]:
        """Диагностика проблем с JSON файлом"""
        try:
            result = {
                'file_exists': os.path.exists(filepath),
                'file_size': 0,
                'encoding': 'unknown',
                'line_count': 0,
                'json_errors': [],
                'sample_lines': []
            }
            
            if not result['file_exists']:
                return result
            
            result['file_size'] = os.path.getsize(filepath)
            
            # Пробуем разные кодировки
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251', 'windows-1251']
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                        result['line_count'] = len(lines)
                        result['encoding'] = encoding
                        result['sample_lines'] = lines[:3]  # Первые 3 строки
                        
                        # Пробуем парсить каждую строку
                        for i, line in enumerate(lines[:10]):  # Проверяем первые 10 строк
                            line = line.strip()
                            if line:
                                try:
                                    json.loads(line)
                                except json.JSONDecodeError as e:
                                    result['json_errors'].append({
                                        'line': i + 1,
                                        'error': str(e),
                                        'preview': line[:100] + '...' if len(line) > 100 else line
                                    })
                        
                        break  # Если успешно прочитали, выходим
                        
                except UnicodeDecodeError:
                    continue
            
            return result
            
        except Exception as e:
            return {'error': f'Диагностика не удалась: {str(e)}'}


    def _extract_from_item(self, item: Any, texts: List[str], labels: List[str]):
        """Извлечение текста и меток из элемента данных"""
        if isinstance(item, dict):
            # Различные возможные ключи для текста
            text_keys = ['text', 'content', 'question', 'input', 'sentence', 'message', 'prompt', 'title', 'body']
            label_keys = ['label', 'category', 'class', 'output', 'answer', 'target', 'sentiment', 'tags', 'type']
            
            text = None
            for key in text_keys:
                if key in item and item[key]:
                    text_value = item[key]
                    if isinstance(text_value, str):
                        text = text_value
                    elif isinstance(text_value, list):
                        text = ' '.join(str(x) for x in text_value)
                    else:
                        text = str(text_value)
                    break
            
            # Если не нашли в стандартных ключах, пробуем найти любую строку
            if text is None:
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:  # Достаточно длинная строка
                        text = value
                        break
            
            label = 'unknown'
            for key in label_keys:
                if key in item and item[key]:
                    label_value = item[key]
                    if isinstance(label_value, (list, dict)):
                        label = str(label_value)
                    else:
                        label = str(label_value)
                    break
            
            if text and isinstance(text, str) and text.strip():
                texts.append(text.strip())
                labels.append(str(label))
        
        elif isinstance(item, str):
            if item.strip():
                texts.append(item.strip())
                labels.append('unknown')
        elif isinstance(item, (list, tuple)):
            # Рекурсивно обрабатываем списки
            for subitem in item:
                self._extract_from_item(subitem, texts, labels)
    
    def _load_csv_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка CSV/TSV датасета"""
        try:
            # Автоопределение разделителя
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = ',' if ',' in first_line else '\t' if '\t' in first_line else ','
            
            df = pd.read_csv(filepath, delimiter=delimiter, encoding='utf-8')
            
            texts = []
            labels = []
            
            # Поиск колонок с текстом и метками
            text_columns = ['text', 'content', 'question', 'input', 'sentence', 'message', 'title', 'body']
            label_columns = ['label', 'category', 'class', 'target', 'sentiment', 'tags', 'type']
            
            text_col = None
            label_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(tc in col_lower for tc in text_columns):
                    text_col = col
                elif any(lc in col_lower for lc in label_columns):
                    label_col = col
            
            # Если не нашли стандартные колонки, используем первую для текста и вторую для меток
            if text_col is None and len(df.columns) > 0:
                text_col = df.columns[0]
            if label_col is None and len(df.columns) > 1:
                label_col = df.columns[1]
            
            if text_col is not None:
                for _, row in df.iterrows():
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    label = str(row[label_col]) if label_col and pd.notna(row.get(label_col, '')) else "unknown"
                    
                    if text.strip():
                        texts.append(text)
                        labels.append(label)
            
            return self._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки CSV датасета: {e}")
            return np.array([]), np.array([])
    
    def _load_excel_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка Excel датасета"""
        try:
            # Загружаем все листы
            excel_file = pd.ExcelFile(filepath)
            all_texts = []
            all_labels = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                
                texts = []
                labels = []
                
                # Поиск колонок с текстом и метками
                text_columns = ['text', 'content', 'question', 'input', 'sentence']
                label_columns = ['label', 'category', 'class', 'target']
                
                text_col = None
                label_col = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(tc in col_lower for tc in text_columns):
                        text_col = col
                    elif any(lc in col_lower for lc in label_columns):
                        label_col = col
                
                # Если не нашли, используем первую колонку для текста
                if text_col is None and len(df.columns) > 0:
                    text_col = df.columns[0]
                if label_col is None and len(df.columns) > 1:
                    label_col = df.columns[1]
                
                if text_col is not None:
                    for _, row in df.iterrows():
                        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                        label = str(row[label_col]) if label_col and pd.notna(row.get(label_col, '')) else f"sheet_{sheet_name}"
                        
                        if text.strip():
                            texts.append(text)
                            labels.append(label)
                
                all_texts.extend(texts)
                all_labels.extend(labels)
            
            return self._prepare_data(all_texts, all_labels)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel датасета: {e}")
            return np.array([]), np.array([])
    
    def _load_text_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка текстового датасета"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разные стратегии разделения текста
            texts = []
            
            # Пробуем разделить по пустым строкам
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Если мало параграфов, пробуем разделить по точкам
            if len(paragraphs) < 5:
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                texts.extend(sentences)
            else:
                texts.extend(paragraphs)
            
            # Создаем метки на основе источника
            labels = ['text_document'] * len(texts)
            
            return self._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки текстового датасета: {e}")
            return np.array([]), np.array([])
    
    def _load_yaml_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка YAML датасета"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            texts = []
            labels = []
            
            # Рекурсивно извлекаем текстовые данные
            self._extract_from_yaml(data, texts, labels)
            
            return self._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки YAML датасета: {e}")
            return np.array([]), np.array([])
    
    def _extract_from_yaml(self, data: Any, texts: List[str], labels: List[str], current_label: str = "yaml"):
        """Рекурсивное извлечение данных из YAML"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 10:  # Только достаточно длинные строки
                    texts.append(value)
                    labels.append(f"{current_label}_{key}")
                else:
                    self._extract_from_yaml(value, texts, labels, f"{current_label}_{key}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and len(item) > 10:
                    texts.append(item)
                    labels.append(f"{current_label}_{i}")
                else:
                    self._extract_from_yaml(item, texts, labels, f"{current_label}_{i}")
    
    def _load_xml_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка XML датасета"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            texts = []
            labels = []
            
            # Рекурсивно обходим XML дерево
            self._extract_from_xml(root, texts, labels)
            
            return self._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки XML датасета: {e}")
            return np.array([]), np.array([])
    
    def _extract_from_xml(self, element: ET.Element, texts: List[str], labels: List[str]):
        """Извлечение текста из XML элемента"""
        if element.text and element.text.strip():
            texts.append(element.text.strip())
            labels.append(element.tag)
        
        for child in element:
            self._extract_from_xml(child, texts, labels)
    
    def _load_parquet_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка Parquet датасета"""
        try:
            df = pd.read_parquet(filepath)
            return self._process_dataframe(df)
        except Exception as e:
            logger.error(f"Ошибка загрузки Parquet датасета: {e}")
            return np.array([]), np.array([])
    
    def _load_feather_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка Feather датасета"""
        try:
            df = pd.read_feather(filepath)
            return self._process_dataframe(df)
        except Exception as e:
            logger.error(f"Ошибка загрузки Feather датасета: {e}")
            return np.array([]), np.array([])
    
    def _load_pickle_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка Pickle датасета"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            texts = []
            labels = []
            
            # Обработка различных структур pickle данных
            if isinstance(data, (list, np.ndarray)):
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                        labels.append('pickle_data')
                    elif isinstance(item, (dict, pd.Series)):
                        self._extract_from_item(item, texts, labels)
            elif isinstance(data, dict):
                self._extract_from_item(data, texts, labels)
            elif isinstance(data, pd.DataFrame):
                return self._process_dataframe(data)
            
            return self._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Pickle датасета: {e}")
            return np.array([]), np.array([])
    
    def _process_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Обработка DataFrame для извлечения текстов и меток"""
        texts = []
        labels = []
        
        # Автопоиск колонок
        text_cols = []
        label_cols = []
        
        for col in df.columns:
            col_str = str(col).lower()
            # Проверяем, содержит ли колонка строковые данные
            if df[col].dtype == 'object' and df[col].notna().any():
                sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if isinstance(sample_value, str) and len(sample_value) > 5:
                    if any(keyword in col_str for keyword in ['text', 'content', 'question', 'message']):
                        text_cols.append(col)
                    elif any(keyword in col_str for keyword in ['label', 'category', 'class']):
                        label_cols.append(col)
        
        # Если не нашли явные текстовые колонки, используем все строковые
        if not text_cols:
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].notna().any():
                    sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                    if isinstance(sample_value, str) and len(sample_value) > 5:
                        text_cols.append(col)
        
        # Собираем данные
        for text_col in text_cols[:1]:  # Берем первую подходящую колонку
            label_col = label_cols[0] if label_cols else None
            
            for _, row in df.iterrows():
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                label = str(row[label_col]) if label_col and pd.notna(row.get(label_col, '')) else f"col_{text_col}"
                
                if text.strip():
                    texts.append(text)
                    labels.append(label)
        
        return self._prepare_data(texts, labels)
    
    def _prepare_data(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        if not texts:
            logger.error("Датасет не содержит данных")
            return np.array([]), np.array([])
        
        # Фильтруем пустые тексты
        filtered_texts = []
        filtered_labels = []
        
        for text, label in zip(texts, labels):
            if text and isinstance(text, str) and text.strip():
                filtered_texts.append(text.strip())
                filtered_labels.append(str(label))
        
        if not filtered_texts:
            logger.error("После фильтрации датасет не содержит данных")
            return np.array([]), np.array([])
        
        logger.info(f"Загружено {len(filtered_texts)} примеров, {len(set(filtered_labels))} уникальных меток")
        
        # Векторизация текстов
        vectorizer = SimpleTextVectorizer(max_features=1000)
        X = vectorizer.fit_transform(filtered_texts)
        
        # Преобразование меток в числовой формат
        unique_labels = list(set(filtered_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_to_idx[label] for label in filtered_labels])
        
        return X, y
    
    def get_dataset_info(self, filename: str) -> Dict[str, Any]:
        """Получить информацию о датасете с поддержкой больших файлов"""
        try:
            filepath = os.path.join(self.datasets_dir, filename)
            
            if not os.path.exists(filepath):
                return {"error": "Файл не найден"}
            
            file_format = self.detect_format(filename)
            file_size = os.path.getsize(filepath)
            
            info = {
                "filename": filename,
                "format": file_format,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "size_gb": round(file_size / (1024 * 1024 * 1024), 2),
                "supported": file_format is not None,
                "is_large": file_size > 100 * 1024 * 1024  # Файлы больше 100MB считаем большими
            }
            
            # Для больших файлов пробуем получить базовую информацию без полной загрузки
            if info['is_large']:
                info['large_file'] = True
                info['samples_count'] = 'много (требуется обучение для подсчета)'
                info['loaded_successfully'] = True
            else:
                # Для маленьких файлов загружаем как обычно
                X, y = self.load_dataset(filename)
                
                if len(X) > 0:
                    info.update({
                        "samples_count": len(X),
                        "features_count": X.shape[1] if len(X.shape) > 1 else 0,
                        "classes_count": len(set(y)) if len(y) > 0 else 0,
                        "loaded_successfully": True
                    })
                else:
                    info["loaded_successfully"] = False
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Получить список доступных датасетов с информацией"""
        try:
            files = os.listdir(self.datasets_dir)
            datasets = []
            
            for file in files:
                if any(file.endswith(ext) for ext in self.get_all_supported_extensions()):
                    info = self.get_dataset_info(file)
                    datasets.append(info)
            
            logger.info(f"Найдено датасетов: {len(datasets)}")
            return datasets
            
        except Exception as e:
            logger.error(f"Ошибка получения списка датасетов: {e}")
            return []

# =============================================================================
# ОБНОВЛЕННЫЙ КЛАСС ДЛЯ ОБУЧЕНИЯ С ДАТАСЕТАМИ
# =============================================================================

class AdvancedDatasetTrainer:
    """Продвинутый тренер для работы с различными форматами датасетов"""
    
    def __init__(self):
        self.dataset_loader = AdvancedDatasetLoader()
        self.vectorizer = SimpleTextVectorizer(max_features=1000)
        
    async def download_from_github(self, url: str) -> Optional[str]:
        """Скачать датасет с GitHub"""
        return await self.dataset_loader.download_from_github(url)
    
    def load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка датасета"""
        return self.dataset_loader.load_dataset(filename)
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Получить список доступных датасетов с информацией"""
        return self.dataset_loader.get_available_datasets()
    
    def get_dataset_info(self, filename: str) -> Dict[str, Any]:
        """Получить информацию о датасете"""
        return self.dataset_loader.get_dataset_info(filename)
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Получить список поддерживаемых форматов"""
        return self.dataset_loader.supported_formats
    
    def save_vectorizer(self, filename: str = "dataset_vectorizer.pkl"):
        """Сохранение векторizer'а"""
        self.vectorizer.save(filename)
    
    def load_vectorizer(self, filename: str = "dataset_vectorizer.pkl"):
        """Загрузка векторizer'а"""
        return self.vectorizer.load(filename)

# =============================================================================
# ОБНОВЛЕННЫЙ GitHubDatasetLoader ДЛЯ ПОДДЕРЖКИ НОВЫХ ФОРМАТОВ
# =============================================================================

class GitHubDatasetLoader:
    """Улучшенный загрузчик датасетов с GitHub с поддержкой новых форматов"""
    
    def __init__(self):
        self.datasets_dir = "training_datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; EducationBot/1.0)',
            'Accept': 'application/vnd.github.v3+json'
        })
        self.supported_extensions = AdvancedDatasetLoader().get_all_supported_extensions()
    
    def extract_github_info(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Извлечение информации из GitHub URL"""
        try:
            url = url.strip()
            
            patterns = [
                r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)',
                r'https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)',
                r'https://github\.com/([^/]+)/([^/]+)/raw/([^/]+)/(.+)',
                r'https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)',
                r'https://github\.com/([^/]+)/([^/]+)'
            ]
            
            for pattern in patterns:
                match = re.match(pattern, url)
                if match:
                    groups = match.groups()
                    if len(groups) == 4:
                        owner, repo, branch, filepath = groups
                        return owner, repo, branch, filepath
                    elif len(groups) == 2:
                        owner, repo = groups
                        return owner, repo, 'main', ''
            
            return None, None, None, None
            
        except Exception as e:
            logger.error(f"Ошибка при разборе GitHub URL: {e}")
            return None, None, None, None
    
    async def download_from_github(self, url: str) -> Optional[str]:
        """Скачивание датасета с GitHub"""
        try:
            owner, repo, branch, filepath = self.extract_github_info(url)
            
            if not owner or not repo:
                logger.error("Не удалось извлечь информацию из GitHub URL")
                return None
            
            # Если указан конкретный файл
            if filepath:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}"
                filename = os.path.basename(filepath)
                
                response = self.session.get(raw_url, timeout=30)
                if response.status_code == 200:
                    file_path = os.path.join(self.datasets_dir, filename)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Датасет скачан: {filename}")
                    return filename
                else:
                    logger.error(f"Ошибка скачивания файла: {response.status_code}")
                    return None
            else:
                # Ищем датасеты в репозитории
                return await self.explore_repository(owner, repo, branch)
                
        except Exception as e:
            logger.error(f"Ошибка при скачивании с GitHub: {e}")
            return None
    
    async def explore_repository(self, owner: str, repo: str, branch: str) -> Optional[str]:
        """Поиск датасетов в репозитории"""
        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
            response = self.session.get(api_url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Ошибка API GitHub: {response.status_code}")
                return None
            
            contents = response.json()
            dataset_files = []
            
            for item in contents:
                if isinstance(item, dict):
                    name = item.get('name', '')
                    # Проверяем все поддерживаемые расширения
                    if any(name.endswith(ext) for ext in self.supported_extensions):
                        dataset_files.append(item['name'])
            
            if not dataset_files:
                logger.info("В репозитории не найдены файлы датасетов")
                return None
            
            # Скачиваем первый найденный датасет
            if dataset_files:
                dataset_file = dataset_files[0]
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{dataset_file}"
                
                response = self.session.get(raw_url, timeout=30)
                if response.status_code == 200:
                    file_path = os.path.join(self.datasets_dir, dataset_file)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Датасет скачан: {dataset_file}")
                    return dataset_file
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при исследовании репозитория: {e}")
            return None
    
    def get_supported_formats(self) -> List[str]:
        """Получить список поддерживаемых форматов"""
        formats = []
        for format_type, extensions in AdvancedDatasetLoader().supported_formats.items():
            formats.append(f"{format_type.upper()} ({', '.join(extensions)})")
        return formats

# =============================================================================
# КЛАСС SELF LEARNING AI
# =============================================================================

class NeuralNetwork(nn.Module):
    """Гибкая нейронная сеть с настраиваемой архитектурой"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 dropout_rate: float = 0.3):
        super(NeuralNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        
        # Входной слой
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            if hidden_size > 1:  # BatchNorm требует минимум 2 элемента в батче
                self.layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Выходной слой
        self.output_layer = nn.Linear(prev_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
        # Инициализация весов
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Инициализация весов сети"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x):
        # Убеждаемся, что вход имеет правильную размерность
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Проверяем размерность входа
        if x.shape[1] != self.input_size:
            # Автоматически изменяем размер если нужно
            if x.shape[1] > self.input_size:
                x = x[:, :self.input_size]
            else:
                # Дополняем нулями если размер меньше
                padding = torch.zeros(x.shape[0], self.input_size - x.shape[1])
                x = torch.cat([x, padding], dim=1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return self.softmax(x)

class ExperienceReplay:
    """Буфер воспроизведения опыта для обучения с подкреплением"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Добавление опыта в буфер"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> list:
        """Выборка случайного батча из буфер"""
        if len(self.buffer) < batch_size:
            return []
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class LearningDataset(Dataset):
    """Датасет для обучения нейросети"""
    
    def __init__(self, data: List[tuple]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, target = self.data[idx]
        return torch.FloatTensor(state), torch.FloatTensor(target)

class SelfLearningAI:
    """Нейросеть с возможностями самообучения"""
    
    def __init__(self, input_size: int = 100, output_size: int = 10, 
                 hidden_sizes: List[int] = None, learning_rate: float = 0.001):
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Инициализация нейросети
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(input_size, hidden_sizes, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Система самообучения
        self.experience_replay = ExperienceReplay()
        self.learning_history = []
        self.performance_metrics = {
            'accuracy': [],
            'loss': [],
            'reward': []
        }
        
        # Активное обучение
        self.uncertainty_threshold = 0.3
        self.learning_rate_decay = 0.95
        
        # Обучение с датасетами
        self.dataset_trainer = AdvancedDatasetTrainer()
        
        # Менеджер датасетов
        self.dataset_manager = DatasetManager()
        
        # Загрузка предыдущего обучения
        self.load_model()
        
        logger.info(f"SelfLearningAI инициализирован на устройстве: {self.device}")
    
    def load_model(self, filepath: str = "self_learning_model.pth"):
        """Загрузка модели и данных обучения"""
        try:
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath, map_location=self.device)
                
                # Проверяем совместимость архитектуры
                if (checkpoint.get('input_size') == self.input_size and 
                    checkpoint.get('output_size') == self.output_size):
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.performance_metrics = checkpoint.get('performance_metrics', {'accuracy': [], 'loss': [], 'reward': []})
                    self.learning_history = checkpoint.get('learning_history', [])
                    
                    # Восстановление буфера опыта
                    experience_buffer = checkpoint.get('experience_buffer', [])
                    self.experience_replay.buffer = deque(experience_buffer, maxlen=10000)
                    
                    logger.info(f"Модель загружена: {filepath}")
                    return True
                else:
                    logger.warning("Архитектура модели не совпадает, пропускаем загрузку")
                    return False
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель: {e}")
        return False
    
    def _initialize_model(self, input_size: int, output_size: int):
        """Инициализация модели с заданными размерами"""
        hidden_sizes = [128, 64, 32]
        self.model = NeuralNetwork(input_size, hidden_sizes, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.input_size = input_size
        self.output_size = output_size

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Предсказание на основе входных данных"""
        if len(input_data) == 0:
            return np.array([])
            
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).to(self.device)
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0)
            output = self.model(input_tensor)
            return output.cpu().numpy()
    
    def predict_with_confidence(self, input_data: np.ndarray) -> tuple:
        """Предсказание с оценкой уверенности"""
        predictions = self.predict(input_data)
        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])
        confidence = np.max(predictions, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class, confidence, predictions
    
    def learn_from_data(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, 
                batch_size: int = 32, validation_data: tuple = None):
        """Обучение на размеченных данных с исправленной обработкой ошибок"""
        
        # ЯВНАЯ ПРОВЕРКА ДАННЫХ ВМЕСТО ТИХОГО ПРЕРЫВАНИЯ
        if len(X) == 0 or len(y) == 0:
            error_msg = f"❌ Пустые данные для обучения: X={len(X)}, y={len(y)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ПРОВЕРКА СОВПАДЕНИЯ РАЗМЕРОВ
        if len(X) != len(y):
            error_msg = f"❌ Несовпадение размеров X({len(X)}) и y({len(y)})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
        logger.info(f"🚀 Начало обучения: {len(X)} примеров, {len(np.unique(y))} классов")
        logger.info(f"📊 Формы данных - X: {X.shape}, y: {y.shape}")
        logger.info(f"🎯 Уникальные классы в y: {np.unique(y)}")
        
        try:
            # ПРОВЕРЯЕМ И ОБНОВЛЯЕМ РАЗМЕР ВХОДНОГО СЛОЯ ЕСЛИ НУЖНО
            if X.shape[1] != self.input_size:
                logger.info(f"🔄 Обновляем размер входного слоя с {self.input_size} до {X.shape[1]}")
                self._update_input_size(X.shape[1])
            
            # ПРЕОБРАЗУЕМ y В ONE-HOT ЕСЛИ НУЖНО
            y_onehot = y
            if len(y.shape) == 1:
                try:
                    # Убеждаемся, что все метки в допустимом диапазоне
                    unique_classes = len(np.unique(y))
                    if unique_classes > self.output_size:
                        logger.info(f"🔄 Обновляем размер выходного слоя с {self.output_size} до {unique_classes}")
                        self._update_output_size(unique_classes)
                    
                    # Преобразуем в one-hot encoding
                    y_onehot = np.eye(self.output_size)[y]
                    logger.info(f"✅ One-hot encoded y: {y_onehot.shape}")
                    
                except Exception as e:
                    error_msg = f"❌ Ошибка преобразования y в one-hot: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # ФИНАЛЬНАЯ ПРОВЕРКА РАЗМЕРНОСТЕЙ
            if len(X) != len(y_onehot):
                error_msg = f"❌ Несовпадение размеров после преобразования: X={len(X)}, y_onehot={len(y_onehot)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # СОЗДАЕМ ДАТАСЕТ И DATALOADER
            # ИСПРАВЛЕНИЕ 1: Преобразуем данные в правильный тип
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y_onehot)
            dataset = LearningDataset(list(zip(X_tensor, y_tensor)))
            
            # Убеждаемся, что batch_size не больше размера датасета
            actual_batch_size = min(batch_size, len(X))
            if actual_batch_size == 0:
                error_msg = "❌ Размер батча равен 0"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # ИСПРАВЛЕНИЕ 2: Убираем drop_last=True для маленьких датасетов
            drop_last = len(X) > actual_batch_size
            dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=drop_last)
            
            self.model.train()
            epoch_losses = []
            
            logger.info(f"🔥 Запуск обучения на {epochs} эпох, batch_size={actual_batch_size}")
            
            for epoch in range(epochs):
                total_loss = 0
                batch_count = 0
                
                for batch_X, batch_y in dataloader:
                    if len(batch_X) == 0:
                        logger.warning("⚠️ Пустой батч, пропускаем")
                        continue
                    
                    # ПРОВЕРЯЕМ РАЗМЕРНОСТИ
                    if batch_X.dim() != 2:
                        logger.warning(f"🔄 Исправляем размерность batch_X: {batch_X.shape}")
                        batch_X = batch_X.view(batch_X.size(0), -1)
                    
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА РАЗМЕРНОСТЕЙ
                    if batch_X.shape[1] != self.input_size:
                        # ИСПРАВЛЕНИЕ 3: Автоматическое приведение размеров вместо пропуска
                        if batch_X.shape[1] > self.input_size:
                            batch_X = batch_X[:, :self.input_size]
                        else:
                            # Дополняем нулями
                            padding = torch.zeros(batch_X.shape[0], self.input_size - batch_X.shape[1], device=self.device)
                            batch_X = torch.cat([batch_X, padding], dim=1)
                        logger.warning(f"🔄 Исправлен размер батча: {batch_X.shape}")
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    
                    # ИСПРАВЛЕНИЕ 4: Проверка выходов модели
                    if torch.isnan(outputs).any():
                        logger.error("❌ Выходы модели содержат NaN значения")
                        continue
                    
                    loss = self.criterion(outputs, batch_y)
                    
                    # ИСПРАВЛЕНИЕ 5: Проверка значения потерь
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error("❌ Потери содержат NaN или Inf значения")
                        continue
                    
                    loss.backward()
                    
                    # ИСПРАВЛЕНИЕ 6: Градиентный clipping для стабильности
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                    epoch_losses.append(avg_loss)
                    
                    # ВАЛИДАЦИЯ
                    val_accuracy = 0
                    if validation_data:
                        val_accuracy = self.evaluate(*validation_data)
                        logger.info(f"📈 Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                    else:
                        logger.info(f"📈 Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                else:
                    logger.warning(f"⚠️ Эпоха {epoch+1}: нет данных для обучения")
                    # ИСПРАВЛЕНИЕ 7: Добавляем значение потерь даже при пустой эпохе
                    epoch_losses.append(float('inf'))
            
            # СОХРАНЕНИЕ МЕТРИК
            self.performance_metrics['loss'].extend(epoch_losses)
            if validation_data:
                self.performance_metrics['accuracy'].append(val_accuracy)
            
            # СОХРАНЕНИЕ МОДЕЛИ
            self.save_model()
            
            # ИСПРАВЛЕНИЕ 8: Безопасное получение финального loss
            final_loss = epoch_losses[-1] if epoch_losses and epoch_losses[-1] != float('inf') else 'N/A'
            logger.info(f"✅ Обучение завершено успешно! Финальный loss: {final_loss}")
            
            return epoch_losses
            
        except Exception as e:
            error_msg = f"❌ Критическая ошибка в learn_from_data: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Трассировка ошибки: {traceback.format_exc()}")
            raise

    def _update_input_size(self, new_input_size: int):
        """Обновление размера входного слоя"""
        try:
            # Сохраняем старые веса если возможно
            old_weights = None
            old_bias = None
            if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                for layer in self.model.layers:
                    if isinstance(layer, nn.Linear):
                        old_weights = layer.weight.data.cpu().numpy()
                        old_bias = layer.bias.data.cpu().numpy()
                        break
            
            # Создаем новую модель с правильным входным размером
            hidden_sizes = []
            if hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    if isinstance(layer, nn.Linear):
                        hidden_sizes.append(layer.out_features)
            
            # Убираем выходной слой из hidden_sizes
            if hidden_sizes and hidden_sizes[-1] == self.output_size:
                hidden_sizes = hidden_sizes[:-1]
            
            if not hidden_sizes:
                hidden_sizes = [128, 64, 32]  # значения по умолчанию
            
            # Пересоздаем модель
            self.model = NeuralNetwork(
                input_size=new_input_size,
                hidden_sizes=hidden_sizes,
                output_size=self.output_size
            ).to(self.device)
            
            # Инициализируем веса
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            self.input_size = new_input_size
            logger.info(f"Размер входного слоя обновлен до {new_input_size}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления входного слоя: {e}")
            # Создаем полностью новую модель в случае ошибки
            self.model = NeuralNetwork(
                input_size=new_input_size,
                hidden_sizes=[128, 64, 32],
                output_size=self.output_size
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.input_size = new_input_size

    def _update_output_size(self, new_output_size: int):
        """Обновление размера выходного слоя"""
        try:
            # Сохраняем старые веса
            old_weights = self.model.output_layer.weight.data.cpu().numpy()
            old_bias = self.model.output_layer.bias.data.cpu().numpy()
            
            # Создаем новый выходной слой
            in_features = self.model.output_layer.in_features
            self.model.output_layer = nn.Linear(in_features, new_output_size).to(self.device)
            self.output_size = new_output_size
            
            # Инициализируем новый слой
            nn.init.xavier_uniform_(self.model.output_layer.weight)
            if self.model.output_layer.bias is not None:
                self.model.output_layer.bias.data.zero_()
            
            # Копируем старые веса если возможно
            if (old_weights.shape[0] <= new_output_size and 
                old_weights.shape[1] == in_features and
                old_bias.shape[0] <= new_output_size):
                
                self.model.output_layer.weight.data[:old_weights.shape[0]] = torch.FloatTensor(old_weights).to(self.device)
                self.model.output_layer.bias.data[:old_bias.shape[0]] = torch.FloatTensor(old_bias).to(self.device)
            
            # Обновляем optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'])
            
            logger.info(f"Размер выходного слоя обновлен до {new_output_size}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления выходного слоя: {e}")
            # Полная пересоздание модели как запасной вариант
            self._initialize_model(self.input_size, new_output_size)

    def learn_from_dataset(self, dataset_filename: str, epochs: int = 5) -> bool:
        """Обучение на загруженном датасете с поддержкой больших файлов"""
        try:
            logger.info(f"🔄 Запуск обучения на датасете: {dataset_filename}")
            
            # Проверяем размер файла перед загрузкой
            filepath = os.path.join("training_datasets", dataset_filename)
            if not os.path.exists(filepath):
                logger.error(f"Файл датасета не найден: {filepath}")
                return False
            
            file_size = os.path.getsize(filepath)
            logger.info(f"Размер файла датасета: {file_size / (1024*1024):.2f}MB")
            
            # Для больших файлов используем оптимизированную загрузку
            if file_size > 100 * 1024 * 1024:  # Для файлов больше 100MB
                logger.info("Используем оптимизированную загрузку для большого датасета")
                X, y = self._load_large_dataset(filepath)
            else:
                X, y = self.dataset_trainer.load_dataset(dataset_filename)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("Не удалось загрузить данные из датасета или данные пустые")
                return False
            
            logger.info(f"Загружено {len(X)} примеров, {len(np.unique(y))} классов")
            logger.info(f"Размерность X: {X.shape}")
            
            # Проверяем и обновляем размеры модели
            if X.shape[1] != self.input_size:
                logger.info(f"Обновляем входной размер с {self.input_size} до {X.shape[1]}")
                self._update_input_size(X.shape[1])
            
            unique_classes = len(np.unique(y))
            if unique_classes > self.output_size:
                logger.info(f"Обновляем размер выходного слоя с {self.output_size} до {unique_classes}")
                self._update_output_size(unique_classes)
            
            # Для больших датасетов уменьшаем batch_size и увеличиваем patience
            batch_size = 8 if len(X) > 10000 else 16
            actual_epochs = min(epochs, 10) if len(X) > 50000 else epochs
            
            logger.info(f"Используемые параметры: batch_size={batch_size}, epochs={actual_epochs}")
            
            # Обучаем модель
            losses = self.learn_from_data(X, y, epochs=actual_epochs, batch_size=batch_size)
            
            if not losses:
                logger.error("Обучение не дало результатов")
                return False
            
            # Сохраняем векторizer
            self.dataset_trainer.save_vectorizer("dataset_vectorizer.pkl")
            
            # Сохраняем модель после успешного обучения
            self.save_model()
            
            logger.info("✅ Обучение на датасете завершено успешно!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении на датасете: {e}")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
            return False

    def _load_large_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Оптимизированная загрузка больших датасетов"""
        try:
            file_extension = os.path.splitext(filepath)[1].lower()
            
            if file_extension in ['.csv', '.txt', '.tsv']:
                # Для CSV файлов используем pandas с chunking
                return self._load_large_csv(filepath)
            elif file_extension in ['.json', '.jsonl']:
                # Для JSON файлов используем потоковую загрузку
                return self._load_large_json(filepath)
            else:
                # Для других форматов используем стандартную загрузку
                filename = os.path.basename(filepath)
                return self.dataset_trainer.load_dataset(filename)
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке большого датасета: {e}")
            return np.array([]), np.array([])

    def _load_large_csv(self, filepath: str, chunk_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка больших CSV файлов по частям"""
        try:
            all_texts = []
            all_labels = []
            
            # Определяем разделитель
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = ',' if ',' in first_line else '\t' if '\t' in first_line else ','
            
            # Загружаем данные по частям
            for chunk in pd.read_csv(filepath, delimiter=delimiter, encoding='utf-8', chunksize=chunk_size):
                texts = []
                labels = []
                
                # Поиск колонок с текстом и метками
                text_cols = [col for col in chunk.columns if any(keyword in col.lower() for keyword in ['text', 'content', 'question'])]
                label_cols = [col for col in chunk.columns if any(keyword in col.lower() for keyword in ['label', 'category', 'class'])]
                
                text_col = text_cols[0] if text_cols else chunk.columns[0]
                label_col = label_cols[0] if label_cols else (chunk.columns[1] if len(chunk.columns) > 1 else None)
                
                for _, row in chunk.iterrows():
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    label = str(row[label_col]) if label_col and pd.notna(row.get(label_col, '')) else "unknown"
                    
                    if text.strip():
                        texts.append(text.strip())
                        labels.append(label)
                
                all_texts.extend(texts)
                all_labels.extend(labels)
                
                logger.info(f"Загружено {len(all_texts)} примеров...")
            
            return self.dataset_trainer.vectorizer._prepare_data(all_texts, all_labels)
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке большого CSV: {e}")
            return np.array([]), np.array([])

    def _load_large_json(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка больших JSON файлов с потоковым парсингом"""
        try:
            texts = []
            labels = []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        self.dataset_trainer.dataset_loader._extract_from_item(item, texts, labels)
                        
                        # Логируем прогресс каждые 10000 строк
                        if i > 0 and i % 10000 == 0:
                            logger.info(f"Обработано {i} строк JSON...")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ошибка парсинга JSON строки {i}: {e}")
                        continue
            
            return self.dataset_trainer.vectorizer._prepare_data(texts, labels)
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке большого JSON: {e}")
            return np.array([]), np.array([])

    async def download_and_train_from_github(self, github_url: str, epochs: int = 5) -> bool:
        """Скачать датасет с GitHub и обучить на нем"""
        try:
            logger.info(f"🔄 Скачивание датасета с GitHub: {github_url}")
            
            # Скачиваем датасет
            dataset_filename = await self.dataset_trainer.download_from_github(github_url)
            
            if not dataset_filename:
                logger.error("Не удалось скачать датасет с GitHub")
                return False
            
            logger.info(f"✅ Датасет скачан: {dataset_filename}")
            
            # Обучаем на скачанном датасете
            success = self.learn_from_dataset(dataset_filename, epochs)
            return success
            
        except Exception as e:
            logger.error(f"❌ Ошибка при скачивании и обучении с GitHub: {e}")
            return False
    
    def dataset_predict(self, text: str):
        """Предсказание категории текста с помощью обученной модели"""
        try:
            # Загружаем векторizer
            if not self.dataset_trainer.load_vectorizer("dataset_vectorizer.pkl"):
                logger.error("Не удалось загрузить векторizer датасета")
                return "неизвестно", 0.0
            
            # Векторизация текста
            text_vector = self.dataset_trainer.vectorizer.transform([text])
            
            if len(text_vector) == 0:
                return "неизвестно", 0.0
            
            prediction = self.predict(text_vector)
            if len(prediction) == 0:
                return "неизвестно", 0.0
                
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # В реальной реализации здесь должно быть преобразование индекса в название класса
            category = f"класс_{predicted_class}"
            
            return category, confidence
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании датасета: {e}")
            return "неизвестно", 0.0
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Оценка точности модели"""
        if len(X_test) == 0:
            return 0.0
            
        self.model.eval()
        with torch.no_grad():
            predictions = self.predict(X_test)
            if len(predictions) == 0:
                return 0.0
                
            predicted_classes = np.argmax(predictions, axis=1)
            
            if len(y_test.shape) > 1:
                true_classes = np.argmax(y_test, axis=1)
            else:
                true_classes = y_test
                
            accuracy = np.mean(predicted_classes == true_classes)
            return accuracy
    
    def save_model(self, filepath: str = "self_learning_model.pth"):
        """Сохранение модели и данных обучения"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'performance_metrics': self.performance_metrics,
                'learning_history': self.learning_history,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'experience_buffer': list(self.experience_replay.buffer)
            }
            
            torch.save(checkpoint, filepath)
            
            # Сохранение дополнительной информации
            model_info = {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'last_updated': datetime.now().isoformat(),
                'total_experiences': len(self.experience_replay)
            }
            
            with open('model_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Модель сохранена: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")

# =============================================================================
# ОБНОВЛЕННЫЙ КЛАСС SelfLearningAI С САМООБУЧЕНИЕМ ИЗ ДИАЛОГОВ
# =============================================================================

class EnhancedSelfLearningAI(SelfLearningAI):
    """Улучшенный SelfLearningAI с самообучением из диалогов пользователей"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dialogue_storage = "user_dialogues"
        os.makedirs(self.dialogue_storage, exist_ok=True)
        self.learning_from_dialogues = True
        self.min_dialogue_length = 3  # Минимальная длина диалога для обучения
        self.learning_batch_size = 10  # Размер батча для обучения из диалогов
        
    def save_user_dialogue(self, user_id: int, messages: List[Dict[str, str]]):
        """Сохранить диалог пользователя для обучения"""
        try:
            if len(messages) < self.min_dialogue_length:
                return
                
            dialogue_file = os.path.join(self.dialogue_storage, f"user_{user_id}.jsonl")
            
            # Сохраняем только последние N сообщений чтобы избежать переполнения
            recent_messages = messages[-20:]  # Последние 20 сообщений
            
            with open(dialogue_file, 'a', encoding='utf-8') as f:
                for msg in recent_messages:
                    if msg.get('role') and msg.get('content'):
                        f.write(json.dumps({
                            'user_id': user_id,
                            'role': msg['role'],
                            'content': msg['content'],
                            'timestamp': datetime.now().isoformat()
                        }, ensure_ascii=False) + '\n')
                        
            logger.info(f"Сохранен диалог пользователя {user_id}, сообщений: {len(recent_messages)}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения диалога: {e}")
    
    async def learn_from_user_dialogues(self, user_id: int = None):
        """Самообучение из сохраненных диалогов пользователей"""
        try:
            if not self.learning_from_dialogues:
                return False, "Обучение из диалогов отключено"
            
            # Собираем все диалоги
            all_dialogues = []
            for filename in os.listdir(self.dialogue_storage):
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(self.dialogue_storage, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    dialogue = json.loads(line.strip())
                                    if user_id is None or dialogue.get('user_id') == user_id:
                                        all_dialogues.append(dialogue)
                    except Exception as e:
                        logger.error(f"Ошибка чтения файла диалога {filename}: {e}")
            
            if len(all_dialogues) < self.min_dialogue_length:
                return False, f"Недостаточно данных для обучения. Нужно минимум {self.min_dialogue_length} сообщений"
            
            # Группируем диалоги по пользователям и сессиям
            training_data = self._prepare_dialogue_training_data(all_dialogues)
            
            if not training_data:
                return False, "Не удалось подготовить данные для обучения"
            
            # Обучаем модель
            success = self._train_on_dialogues(training_data)
            
            if success:
                # Очищаем старые диалоги после успешного обучения
                self._cleanup_old_dialogues()
                return True, f"✅ Самообучение завершено! Обработано {len(training_data)} диалогов"
            else:
                return False, "❌ Ошибка при обучении на диалогах"
                
        except Exception as e:
            logger.error(f"Ошибка самообучения из диалогов: {e}")
            return False, f"❌ Ошибка самообучения: {str(e)}"
    
    def _prepare_dialogue_training_data(self, dialogues: List[Dict]) -> List[Tuple[str, str]]:
        """Подготовка данных для обучения из диалогов"""
        training_pairs = []
        
        # Группируем по пользователям и временным меткам
        user_sessions = {}
        for dialogue in dialogues:
            user_id = dialogue.get('user_id')
            timestamp = dialogue.get('timestamp', '')
            date_key = timestamp[:10]  # Группируем по дате
            
            key = f"{user_id}_{date_key}"
            if key not in user_sessions:
                user_sessions[key] = []
            
            user_sessions[key].append(dialogue)
        
        # Создаем пары вопрос-ответ из последовательных сообщений
        for session in user_sessions.values():
            session.sort(key=lambda x: x.get('timestamp', ''))
            
            # Проходим по диалогу и создаем пары
            for i in range(len(session) - 1):
                current_msg = session[i]
                next_msg = session[i + 1]
                
                if (current_msg.get('role') == 'user' and 
                    next_msg.get('role') == 'assistant' and
                    len(current_msg.get('content', '')) > 5 and
                    len(next_msg.get('content', '')) > 5):
                    
                    training_pairs.append((
                        current_msg['content'],
                        next_msg['content']
                    ))
        
        return training_pairs
    
    def _train_on_dialogues(self, training_pairs: List[Tuple[str, str]]) -> bool:
        """Обучение на парах вопрос-ответ из диалогов"""
        try:
            if not training_pairs:
                return False
            
            # Подготавливаем данные
            questions = [pair[0] for pair in training_pairs]
            answers = [pair[1] for pair in training_pairs]
            
            # Векторизация
            vectorizer = SimpleTextVectorizer(max_features=1000)
            X = vectorizer.fit_transform(questions)
            
            # Для ответов используем ту же векторзацию или создаем отдельную
            y_vectorizer = SimpleTextVectorizer(max_features=1000)
            y = y_vectorizer.fit_transform(answers)
            
            # Проверяем размерности
            if X.shape[0] == 0 or y.shape[0] == 0:
                return False
            
            # Обучаем модель
            losses = self.learn_from_data(X, y, epochs=3, batch_size=min(8, len(X)))
            
            if losses:
                self.save_model()
                logger.info(f"Самообучение завершено. Обработано {len(training_pairs)} пар")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обучения на диалогах: {e}")
            return False
    
    def _cleanup_old_dialogues(self, days_old: int = 7):
        """Очистка старых диалогов"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for filename in os.listdir(self.dialogue_storage):
                filepath = os.path.join(self.dialogue_storage, filename)
                
                # Для файлов проверяем время изменения
                if os.path.getmtime(filepath) < cutoff_date:
                    os.remove(filepath)
                    logger.info(f"Удален старый файл диалогов: {filename}")
                    
        except Exception as e:
            logger.error(f"Ошибка очистки старых диалогов: {e}")

# =============================================================================
# КЛАСС AdvancedAIAssistant
# =============================================================================

class AIConfig:
    """Конфигурация для ИИ-помощника"""
    
    def __init__(self, api_key: str = "", models: List[str] = None, default_model_index: int = 0):
        self.api_key = api_key
        self.models = models or []
        self.default_model_index = default_model_index
        self.max_tokens = 300
        self.temperature = 0.7
        self.max_history = 6
        self.timeout = 30

# =============================================================================
# ОБНОВЛЕННЫЙ КЛАСС AdvancedAIAssistant С САМООБУЧЕНИЕМ
# =============================================================================

class EnhancedAIAssistant:
    """Улучшенный AI Assistant с самообучением из диалогов"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.current_model_index = config.default_model_index
        self.conversation_cache = {}
        
        # Заменяем SelfLearningAI на улучшенную версию
        self.self_learning_ai = EnhancedSelfLearningAI(
            input_size=100,
            output_size=10,
            hidden_sizes=[256, 128, 64],
            learning_rate=0.001
        )
        
        self.last_learning_time = datetime.now()
        self.learning_interval_hours = 24  # Обучение раз в 24 часа
        
        # Инициализация менеджера датасетов
        self.dataset_manager = DatasetManager()
    
    def _check_and_perform_learning(self):
        """Проверить и выполнить самообучение если нужно"""
        try:
            time_since_last_learning = datetime.now() - self.last_learning_time
            hours_passed = time_since_last_learning.total_seconds() / 3600
            
            if hours_passed >= self.learning_interval_hours:
                logger.info("Запуск периодического самообучения из диалогов...")
                
                # Используем asyncio для запуска в отдельном потоке
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success, message = loop.run_until_complete(
                        self.self_learning_ai.learn_from_user_dialogues()
                    )
                    
                    if success:
                        logger.info("Периодическое самообучение завершено успешно")
                    else:
                        logger.warning(f"Периодическое самообучение не удалось: {message}")
                    
                    self.last_learning_time = datetime.now()
                finally:
                    loop.close()
                
        except Exception as e:
            logger.error(f"Ошибка при проверке самообучения: {e}")

    def get_conversation_history(self, user_id: int) -> List[Dict]:
        """Получить истории разговора для пользователя"""
        if user_id not in self.conversation_cache:
            self.conversation_cache[user_id] = []
        return self.conversation_cache[user_id]
    
    def update_conversation_history(self, user_id: int, user_message: str, ai_response: str):
        """Обновить историю разговора"""
        history = self.get_conversation_history(user_id)
        
        history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response}
        ])
        
        # Ограничиваем историю
        if len(history) > self.config.max_history:
            history = history[-(self.config.max_history):]
        
        self.conversation_cache[user_id] = history
    
    def clear_conversation_history(self, user_id: int):
        """Очистить историю разговора для пользователя"""
        if user_id in self.conversation_cache:
            self.conversation_cache[user_id] = []
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """Преобразование текста в числовые признаки"""
        if not text:
            return np.zeros(100)
            
        words = text.lower().split()
        feature_vector = np.zeros(100)  # Фиксированный размер признаков
        
        for i, word in enumerate(words[:100]):  # Ограничиваем длину
            hash_val = hash(word) % 10000 / 10000.0
            feature_vector[i % 100] += hash_val
        
        # Нормализация
        if np.linalg.norm(feature_vector) > 0:
            feature_vector = feature_vector / np.linalg.norm(feature_vector)
        
        return feature_vector
    
    def _generate_self_learning_response(self, user_message: str, history: List[Dict]) -> str:
        """Генерация ответа с помощью SelfLearningAI"""
        try:
            # Пытаемся использовать датасет классификатор для лучшего понимания темы
            dataset_category, dataset_confidence = self.self_learning_ai.dataset_predict(user_message)
            
            conversation_text = " ".join([msg["content"] for msg in history[-3:]] + [user_message])
            features = self._text_to_features(conversation_text)
            
            prediction = self.self_learning_ai.predict(features.reshape(1, -1))
            if len(prediction) == 0:
                return "Я еще учусь отвечать на вопросы. Можете переформулировать ваш запрос?"
                
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            responses = [
                "Я могу помочь вам с учебными вопросами. Что именно вас интересует?",
                "Похоже, у вас вопрос по учебному материалу. Могу ли я помочь?",
                "Я специализируюсь на помощи студентам. Задайте ваш вопрос!",
                "Для лучшей помощи уточните ваш вопрос по конкретному предмету.",
                "Я могу объяснить сложные темы или помочь с подготовкой к экзаменам.",
                "Если у вас есть конкретный вопрос по лекции или практике, я постараюсь помочь.",
                "Могу помочь структурировать информацию или подготовить конспект.",
                "Для решения учебных задач лучше обратиться к конкретным материалам курса.",
                "Я здесь, чтобы помочь с учебными вопросами любого уровня сложности.",
                "Не стесняйтесь задавать вопросы - я постараюсь дать полезный ответ."
            ]
            
            if confidence > 0.3 and predicted_class < len(responses):
                base_response = responses[predicted_class]
            else:
                base_response = "Я еще учусь отвечать на вопросы. Можете переформулировать ваш запрос?"
            
            # Обогащаем ответ на основе ключевых слов
            if any(word in user_message.lower() for word in ['лекция', 'лекции']):
                base_response += "\n\n📚 Вы можете найти лекции в разделе 'Предметы'."
            elif any(word in user_message.lower() for word in ['практическая', 'практика']):
                base_response += "\n\n📝 Практические работы также доступны в разделе 'Предметы'."
            elif any(word in user_message.lower() for word in ['помощь', 'помоги']):
                base_response += "\n\n👤 Для персональной помощи обратитесь к нашему помощнику в соответствующем разделе."
            
            return base_response
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа SelfLearningAI: {e}")
            return "Извините, возникла техническая ошибка. Попробуйте задать вопрос позже."
    
    async def get_ai_response(self, user_id: int, message: str) -> Tuple[str, bool, str]:
        """
        Получить ответ от ИИ с сохранением диалога для обучения
        """
        try:
            # Получаем ответ как обычно
            history = self.get_conversation_history(user_id)
            ai_response = self._generate_self_learning_response(message, history)
            
            # Сохраняем диалог для обучения
            self.self_learning_ai.save_user_dialogue(user_id, history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ai_response}
            ])
            
            # Обновляем историю
            self.update_conversation_history(user_id, message, ai_response)
            
            # Периодическое самообучение
            self._check_and_perform_learning()
            
            return ai_response, True, "EnhancedSelfLearningAI"
            
        except Exception as e:
            logger.error(f"Ошибка в улучшенном ИИ-помощнике: {e}")
            return "❌ Временно проблемы с ИИ-помощником. Попробуйте позже.", False, "Error"
    
    async def force_learning_from_dialogues(self, user_id: int = None) -> Tuple[bool, str]:
        """Принудительное обучение из диалогов"""
        try:
            logger.info("Запуск принудительного обучения из диалогов...")
            success, message = await self.self_learning_ai.learn_from_user_dialogues(user_id)
            return success, message
        except Exception as e:
            error_msg = f"❌ Ошибка принудительного обучения: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def get_stats(self) -> Dict:
        """Получить статистику использования ИИ"""
        total_conversations = len(self.conversation_cache)
        total_messages = sum(len(history) for history in self.conversation_cache.values())
        
        # Проверяем наличие обученной модели
        model_trained = os.path.exists("self_learning_model.pth")
        
        return {
            "current_model": "EnhancedSelfLearningAI",
            "available_models": ["EnhancedSelfLearningAI"],
            "total_users": total_conversations,
            "total_messages": total_messages,
            "model_trained": model_trained,
            "is_configured": True
        }

    async def train_on_dataset(self, dataset_filename: str) -> Tuple[bool, str]:
        """Обучение на датасете с правильной асинхронностью"""
        try:
            logger.info(f"🔄 Запуск обучения на датасете: {dataset_filename}")
            
            # Используем asyncio для запуска в отдельном потоке
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                lambda: self.self_learning_ai.learn_from_dataset(dataset_filename, 5)
            )
            
            if success:
                return True, "✅ Обучение завершено успешно!"
            else:
                return False, "❌ Ошибка при обучении на датасете"
                
        except Exception as e:
            error_message = f"❌ Ошибка при обучении на датасете: {str(e)}"
            logger.error(error_message)
            return False, error_message
                
        
    async def train_from_github(self, github_url: str) -> Tuple[bool, str]:
        """Обучение на датасете с GitHub с возвратом результата и сообщения"""
        try:
            logger.info(f"🔄 Запуск обучения с GitHub: {github_url}")
            success = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.self_learning_ai.download_and_train_from_github(github_url)
            )

            if success:
                message = (
                    "✅ Датасет успешно скачан с GitHub и модель обучена!\n\n"
                    "ИИ теперь обладает новыми знаниями из датасета."
                )
            else:
                message = (
                    "❌ Не удалось скачать или обучить на датасете с GitHub.\n\n"
                    "Проверьте:\n"
                    "• Доступность репозитория\n"
                    "• Наличие файлов .json или .csv\n"
                    "• Формат данных в датасете"
                )
            
            return success, message
            
        except Exception as e:
            error_message = f"❌ Ошибка при обучении с GitHub: {str(e)}"
            logger.error(error_message)
            return False, error_message

    def get_datasets_info(self) -> List[Dict[str, Any]]:
        """Получить информацию о всех датасетах"""
        return self.dataset_manager.get_all_datasets()

    def delete_dataset(self, filename: str) -> Tuple[bool, str]:
        """Удалить датасет с возвратом результата и сообщения"""
        success = self.dataset_manager.delete_dataset(filename)
        
        if success:
            return True, f"✅ Датасет '{filename}' успешно удален!"
        else:
            return False, f"❌ Не удалось удалить датасет '{filename}'"

# =============================================================================
# КЛАССЫ БАЗЫ ДАННЫХ И ОСНОВНОГО БОТА
# =============================================================================

import sqlite3
import os
import logging
from typing import List, Dict, Optional, Tuple, Any

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_name: str = "bot_database.db"):
        self.db_name = db_name
        self.init_database()


    def get_connection(self) -> sqlite3.Connection:
        """Создание соединения с базой данных"""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Инициализация базы данных с поддержкой преподавателей"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Включаем поддержку внешних ключей
        cursor.execute("PRAGMA foreign_keys = ON")
        
        def init_database(self):
            """Инициализация базы данных с поддержкой преподавателей"""
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Включаем поддержку внешних ключей
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Таблица для управления расписанием
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schedule_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ... остальные существующие таблицы ...
            
            conn.commit()
            conn.close()
         # Таблица для управления расписанием (если нужно)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schedule_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Таблица пользователей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Таблица предметов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        ''')
        
        # Таблица преподавателей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teachers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                subject_id INTEGER,
                FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE
            )
        ''')
        
        # Таблица лекций
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lectures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                teacher_id INTEGER,
                number INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE,
                FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE SET NULL,
                UNIQUE(subject_id, teacher_id, number)
            )
        ''')
        
        # Таблица практических работ
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS practices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                teacher_id INTEGER,
                number INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE,
                FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE SET NULL,
                UNIQUE(subject_id, teacher_id, number)
            )
        ''')
        
        # Таблица для полезного контента
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS useful_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Таблица логов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Добавляем недостающие столбцы если они не существуют
        self._add_missing_columns(cursor)
        
        conn.commit()
        conn.close()
    
    def get_all_schedule(self) -> List[Dict[str, Any]]:
        """Получение всех расписаний"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schedule_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('SELECT * FROM schedule_files ORDER BY uploaded_date DESC')
            schedules = [dict(row) for row in cursor.fetchall()]
            return schedules
        except Exception as e:
            logger.error(f"Ошибка при получении расписаний: {e}")
            return []
        finally:
            conn.close()

    def add_schedule(self, title: str, file_path: str) -> Optional[int]:
        """Добавление расписания"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO schedule_files (title, file_path) VALUES (?, ?)',
                (title, file_path)
            )
            schedule_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Добавлено расписание: {title}, ID: {schedule_id}")
            return schedule_id
        except Exception as e:
            logger.error(f"Ошибка при добавлении расписания: {e}")
            return None
        finally:
            conn.close()

    def get_schedule(self, schedule_id: int) -> Optional[Dict[str, Any]]:
        """Получение расписания по ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM schedule_files WHERE id = ?', (schedule_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Ошибка при получении расписания: {e}")
            return None
        finally:
            conn.close()

    def delete_schedule(self, schedule_id: int) -> bool:
        """Удаление расписания"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM schedule_files WHERE id = ?", (schedule_id,))
            success = cursor.rowcount > 0
            conn.commit()
            logger.info(f"Удалено расписание: ID {schedule_id}")
            return success
        except Exception as e:
            logger.error(f"Ошибка при удалении расписания: {e}")
            return False
        finally:
            conn.close()

    def _add_missing_columns(self, cursor: sqlite3.Cursor):
        """Добавление недостающих столбцов в таблицы"""
        tables_columns = {
            'lectures': ['teacher_id'],
            'practices': ['teacher_id']
        }
        
        for table, columns in tables_columns.items():
            for column in columns:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} INTEGER")
                    logger.info(f"Добавлен столбец {column} в таблицу {table}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        logger.warning(f"Не удалось добавить столбец {column} в {table}: {e}")

    def add_user(self, user_id: int, username: str):
        """Добавление пользователя"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT OR REPLACE INTO users (id, username) VALUES (?, ?)',
                (user_id, username)
            )
            conn.commit()
            logger.info(f"Добавлен пользователь: {user_id}, {username}")
        except Exception as e:
            logger.error(f"Ошибка при добавлении пользователя: {e}")
        finally:
            conn.close()

    def add_subject(self, name: str, description: str = "") -> Optional[int]:
        """Добавление предмета"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT OR IGNORE INTO subjects (name, description) VALUES (?, ?)',
                (name, description)
            )
            
            # Получаем ID добавленного предмета
            cursor.execute('SELECT id FROM subjects WHERE name = ?', (name,))
            result = cursor.fetchone()
            subject_id = result['id'] if result else None
            
            conn.commit()
            logger.info(f"Добавлен предмет: {name}, ID: {subject_id}")
            return subject_id
        except Exception as e:
            logger.error(f"Ошибка при добавлении предмета: {e}")
            return None
        finally:
            conn.close()

    def add_teacher(self, name: str, subject_id: int) -> Optional[int]:
        """Добавление преподавателя"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Проверяем существование предмета
            cursor.execute('SELECT id FROM subjects WHERE id = ?', (subject_id,))
            if not cursor.fetchone():
                logger.error(f"Предмет с ID {subject_id} не существует")
                return None
            
            cursor.execute(
                'INSERT INTO teachers (name, subject_id) VALUES (?, ?)',
                (name, subject_id)
            )
            
            teacher_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Добавлен преподаватель: {name}, ID: {teacher_id}")
            return teacher_id
        except Exception as e:
            logger.error(f"Ошибка при добавлении преподавателя: {e}")
            return None
        finally:
            conn.close()

    def add_lecture(self, subject_id: int, number: int, file_path: str, teacher_id: Optional[int] = None) -> bool:
        """Добавление лекции"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Проверяем существование файла
            if not os.path.exists(file_path):
                logger.error(f"Файл не существует: {file_path}")
                return False
            
            # Используем INSERT OR REPLACE для упрощения
            cursor.execute(
                '''INSERT OR REPLACE INTO lectures 
                   (subject_id, teacher_id, number, file_path) 
                   VALUES (?, ?, ?, ?)''',
                (subject_id, teacher_id, number, file_path)
            )
            
            conn.commit()
            logger.info(f"Добавлена лекция: предмет {subject_id}, номер {number}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при добавлении лекции: {e}")
            return False
        finally:
            conn.close()

    def get_subjects_simple(self) -> List[Dict[str, Any]]:
        """Простое получение предметов без сложных JOIN"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id, name FROM subjects ORDER BY name')
            subjects = [dict(row) for row in cursor.fetchall()]
            return subjects
        except Exception as e:
            logger.error(f"Ошибка при получении предметов: {e}")
            return []
        finally:
            conn.close()

    def add_practice(self, subject_id: int, number: int, file_path: str, teacher_id: Optional[int] = None) -> bool:
        """Добавление практической работы"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Проверяем существование файла
            if not os.path.exists(file_path):
                logger.error(f"Файл не существует: {file_path}")
                return False
            
            cursor.execute(
                '''INSERT OR REPLACE INTO practices 
                   (subject_id, teacher_id, number, file_path) 
                   VALUES (?, ?, ?, ?)''',
                (subject_id, teacher_id, number, file_path)
            )
            
            conn.commit()
            logger.info(f"Добавлена практика: предмет {subject_id}, номер {number}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при добавлении практической работы: {e}")
            return False
        finally:
            conn.close()

    def add_useful_content(self, title: str, file_path: str, content_type: str) -> Optional[int]:
        """Добавление полезного контента"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Проверяем существование файла
            if not os.path.exists(file_path):
                logger.error(f"Файл не существует: {file_path}")
                return None
            
            cursor.execute(
                'INSERT INTO useful_content (title, file_path, type) VALUES (?, ?, ?)',
                (title, file_path, content_type)
            )
            
            content_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Добавлен полезный контент: {title}, ID: {content_id}")
            return content_id
        except Exception as e:
            logger.error(f"Ошибка при добавлении полезного контента: {e}")
            return None
        finally:
            conn.close()

    def get_all_subjects(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT s.*, 
                    COALESCE(GROUP_CONCAT(t.name, ', '), 'Нет преподавателей') as teacher_names
                FROM subjects s
                LEFT JOIN teachers t ON s.id = t.subject_id
                GROUP BY s.id
                ORDER BY s.name
            ''')
            
            subjects = []
            for row in cursor.fetchall():
                subject = dict(row)
                # Гарантируем, что teacher_names всегда строка
                subject['teacher_names'] = subject.get('teacher_names', 'Нет преподавателей')
                subjects.append(subject)
            
            return subjects
        except Exception as e:
            logger.error(f"Ошибка при получении предметов: {e}")
            return []
        finally:
            conn.close()

    def get_subject(self, subject_id: int) -> Optional[Dict[str, Any]]:
        """Получение предмета по ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM subjects WHERE id = ?', (subject_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении предмета: {e}")
            return None
        finally:
            conn.close()

    def get_teachers_by_subject(self, subject_id: int) -> List[Dict[str, Any]]:
        """Получение преподавателей по предмету"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'SELECT * FROM teachers WHERE subject_id = ? ORDER BY name',
                (subject_id,)
            )
            
            teachers = [dict(row) for row in cursor.fetchall()]
            return teachers
        except Exception as e:
            logger.error(f"Ошибка при получении преподавателей: {e}")
            return []
        finally:
            conn.close()

    def get_teacher(self, teacher_id: int) -> Optional[Dict[str, Any]]:
        """Получение преподавателя по ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM teachers WHERE id = ?', (teacher_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении преподавателя: {e}")
            return None
        finally:
            conn.close()

    def get_all_useful_content(self) -> List[Dict[str, Any]]:
        """Получение всего полезного контента"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM useful_content ORDER BY created_at DESC')
            content_list = [dict(row) for row in cursor.fetchall()]
            return content_list
        except Exception as e:
            logger.error(f"Ошибка при получении полезного контента: {e}")
            return []
        finally:
            conn.close()

    def get_useful_content_by_id(self, content_id: int) -> Optional[Dict[str, Any]]:
        """Получение полезного контента по ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM useful_content WHERE id = ?', (content_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Ошибка при получении контента {content_id}: {e}")
            return None
        finally:
            conn.close()

    def delete_useful_content(self, content_id: int) -> bool:
        """Удаление полезного контента"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM useful_content WHERE id = ?", (content_id,))
            success = cursor.rowcount > 0
            conn.commit()
            logger.info(f"Удален полезный контент: ID {content_id}")
            return success
        except Exception as e:
            logger.error(f"Ошибка при удалении контента {content_id}: {e}")
            return False
        finally:
            conn.close()

    def add_log_entry(self, level: str, message: str, user_id: Optional[int] = None) -> bool:
        """Добавление записи в лог"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO logs (level, message, user_id) VALUES (?, ?, ?)",
                (level, message, user_id)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка при добавлении лога: {e}")
            return False
        finally:
            conn.close()

    def get_logs(self, limit: int = 100, log_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение логов с фильтрацией по типу"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if log_type:
                cursor.execute(
                    "SELECT * FROM logs WHERE level = ? ORDER BY created_at DESC LIMIT ?",
                    (log_type.upper(), limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM logs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            logs = [dict(row) for row in cursor.fetchall()]
            return logs
        except Exception as e:
            logger.error(f"Ошибка при получении логов: {e}")
            return []
        finally:
            conn.close()

    def get_subject_lectures_count(self, subject_id: int) -> int:
        """Получение количества лекций по предмету"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM lectures WHERE subject_id = ?",
                (subject_id,)
            )
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка при подсчете лекций: {e}")
            return 0
        finally:
            conn.close()

    def get_subject_practices_count(self, subject_id: int) -> int:
        """Получение количества практик по предмету"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM practices WHERE subject_id = ?",
                (subject_id,)
            )
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка при подсчете практик: {e}")
            return 0
        finally:
            conn.close()

    def get_teacher_by_name_and_subject(self, name: str, subject_id: int) -> Optional[Dict[str, Any]]:
        """Получение преподавателя по имени и предмету"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM teachers WHERE name = ? AND subject_id = ?",
                (name, subject_id)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Ошибка при поиске преподавателя: {e}")
            return None
        finally:
            conn.close()

    def update_teacher_subject(self, teacher_id: int, subject_id: int) -> bool:
        """Обновление предмета преподавателя"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "UPDATE teachers SET subject_id = ? WHERE id = ?",
                (subject_id, teacher_id)
            )
            success = cursor.rowcount > 0
            conn.commit()
            return success
        except Exception as e:
            logger.error(f"Ошибка при обновлении преподавателя: {e}")
            return False
        finally:
            conn.close()

    def get_all_teachers(self) -> List[Dict[str, Any]]:
        """Получение всех преподавателей"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT t.*, s.name as subject_name 
                FROM teachers t 
                LEFT JOIN subjects s ON t.subject_id = s.id 
                ORDER BY t.name
            ''')
            teachers = [dict(row) for row in cursor.fetchall()]
            return teachers
        except Exception as e:
            logger.error(f"Ошибка при получении преподавателей: {e}")
            return []
        finally:
            conn.close()

    def search_useful_content(self, query: str) -> List[Dict[str, Any]]:
        """Поиск полезного контента"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM useful_content WHERE title LIKE ? ORDER BY created_at DESC",
                (f'%{query}%',)
            )
            content = [dict(row) for row in cursor.fetchall()]
            return content
        except Exception as e:
            logger.error(f"Ошибка при поиске контента: {e}")
            return []
        finally:
            conn.close()

    def get_recent_content(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение последнего добавленного контента"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM useful_content ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            content = [dict(row) for row in cursor.fetchall()]
            return content
        except Exception as e:
            logger.error(f"Ошибка при получении последнего контента: {e}")
            return []
        finally:
            conn.close()

    def cleanup_orphaned_files(self) -> int:
        """Очистка записей о несуществующих файлах"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Для лекций
            cursor.execute("SELECT id, file_path FROM lectures")
            lectures = cursor.fetchall()
            deleted_lectures = 0
            
            for lecture in lectures:
                if not os.path.exists(lecture['file_path']):
                    cursor.execute("DELETE FROM lectures WHERE id = ?", (lecture['id'],))
                    deleted_lectures += 1
            
            # Для практик
            cursor.execute("SELECT id, file_path FROM practices")
            practices = cursor.fetchall()
            deleted_practices = 0
            
            for practice in practices:
                if not os.path.exists(practice['file_path']):
                    cursor.execute("DELETE FROM practices WHERE id = ?", (practice['id'],))
                    deleted_practices += 1
            
            # Для полезного контента
            cursor.execute("SELECT id, file_path FROM useful_content")
            useful_content = cursor.fetchall()
            deleted_content = 0
            
            for content in useful_content:
                if not os.path.exists(content['file_path']):
                    cursor.execute("DELETE FROM useful_content WHERE id = ?", (content['id'],))
                    deleted_content += 1
            
            conn.commit()
            total_deleted = deleted_lectures + deleted_practices + deleted_content
            logger.info(f"Очищено orphaned-записей: {total_deleted}")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Ошибка при очистке orphaned-файлов: {e}")
            return 0
        finally:
            conn.close()

    def get_lectures(self, subject_id: int, teacher_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Получение лекций по предмету"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if teacher_id:
                cursor.execute(
                    'SELECT * FROM lectures WHERE subject_id = ? AND teacher_id = ? ORDER BY number',
                    (subject_id, teacher_id)
                )
            else:
                cursor.execute(
                    'SELECT * FROM lectures WHERE subject_id = ? ORDER BY number',
                    (subject_id,)
                )
            
            lectures = [dict(row) for row in cursor.fetchall()]
            return lectures
        except Exception as e:
            logger.error(f"Ошибка при получении лекций: {e}")
            return []
        finally:
            conn.close()

    def get_practices(self, subject_id: int, teacher_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Получение практических работ по предмету"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if teacher_id:
                cursor.execute(
                    'SELECT * FROM practices WHERE subject_id = ? AND teacher_id = ? ORDER BY number',
                    (subject_id, teacher_id)
                )
            else:
                cursor.execute(
                    'SELECT * FROM practices WHERE subject_id = ? ORDER BY number',
                    (subject_id,)
                )
            
            practices = [dict(row) for row in cursor.fetchall()]
            return practices
        except Exception as e:
            logger.error(f"Ошибка при получении практик: {e}")
            return []
        finally:
            conn.close()

    def get_lecture(self, subject_id: int, number: int, teacher_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Получение конкретной лекции"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if teacher_id:
                cursor.execute(
                    'SELECT * FROM lectures WHERE subject_id = ? AND teacher_id = ? AND number = ?',
                    (subject_id, teacher_id, number)
                )
            else:
                cursor.execute(
                    'SELECT * FROM lectures WHERE subject_id = ? AND number = ?',
                    (subject_id, number)
                )
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении лекции: {e}")
            return None
        finally:
            conn.close()

    def _check_and_add_missing_columns(self, cursor):
        """Проверка и добавление отсутствующих колонок в таблицы"""
        try:
            # Проверяем существование колонки folder_id в useful_content
            cursor.execute("PRAGMA table_info(useful_content)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'folder_id' not in columns:
                logger.info("Добавляем отсутствующую колонку folder_id в useful_content")
                cursor.execute('ALTER TABLE useful_content ADD COLUMN folder_id INTEGER')
            
            # Проверяем другие возможные отсутствующие колонки
            cursor.execute("PRAGMA table_info(useful_folders)")
            useful_folders_columns = [column[1] for column in cursor.fetchall()]
            
            if 'name' not in useful_folders_columns:
                logger.info("Добавляем отсутствующую колонку name в useful_folders")
                cursor.execute('ALTER TABLE useful_folders ADD COLUMN name TEXT NOT NULL UNIQUE')
                
        except Exception as e:
            logger.error(f"Ошибка при проверке/добавлении колонок: {e}")

    def get_practice(self, subject_id: int, number: int, teacher_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Получение конкретной практической работы"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if teacher_id:
                cursor.execute(
                    'SELECT * FROM practices WHERE subject_id = ? AND teacher_id = ? AND number = ?',
                    (subject_id, teacher_id, number)
                )
            else:
                cursor.execute(
                    'SELECT * FROM practices WHERE subject_id = ? AND number = ?',
                    (subject_id, number)
                )
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении практики: {e}")
            return None
        finally:
            conn.close()

    def get_useful_content(self, content_id: int) -> Optional[Dict[str, Any]]:
        """Получение полезного контента по ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM useful_content WHERE id = ?', (content_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении контента: {e}")
            return None
        finally:
            conn.close()

    def get_all_useful_content(self) -> List[Dict[str, Any]]:
        """Получение всего полезного контента"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM useful_content ORDER BY id DESC')
            content_list = [dict(row) for row in cursor.fetchall()]
            return content_list
        except Exception as e:
            logger.error(f"Ошибка при получении контента: {e}")
            return []
        finally:
            conn.close()

    def delete_lecture(self, subject_id: int, number: int, teacher_id: Optional[int] = None) -> bool:
        """Удалить лекцию"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if teacher_id:
                cursor.execute(
                    "DELETE FROM lectures WHERE subject_id = ? AND number = ? AND teacher_id = ?",
                    (subject_id, number, teacher_id)
                )
            else:
                cursor.execute(
                    "DELETE FROM lectures WHERE subject_id = ? AND number = ? AND (teacher_id IS NULL OR teacher_id = '')",
                    (subject_id, number)
                )
            
            success = cursor.rowcount > 0
            conn.commit()
            logger.info(f"Удалена лекция: предмет {subject_id}, номер {number}")
            return success
        except Exception as e:
            logger.error(f"Ошибка при удалении лекции: {e}")
            return False
        finally:
            conn.close()

    def delete_practice(self, subject_id: int, number: int, teacher_id: Optional[int] = None) -> bool:
        """Удалить практическую работу"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if teacher_id:
                cursor.execute(
                    "DELETE FROM practices WHERE subject_id = ? AND number = ? AND teacher_id = ?",
                    (subject_id, number, teacher_id)
                )
            else:
                cursor.execute(
                    "DELETE FROM practices WHERE subject_id = ? AND number = ? AND (teacher_id IS NULL OR teacher_id = '')",
                    (subject_id, number)
                )
            
            success = cursor.rowcount > 0
            conn.commit()
            logger.info(f"Удалена практика: предмет {subject_id}, номер {number}")
            return success
        except Exception as e:
            logger.error(f"Ошибка при удалении практики: {e}")
            return False
        finally:
            conn.close()

    def delete_useful_content(self, content_id: int) -> bool:
        """Удаление полезного контента"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM useful_content WHERE id = ?", (content_id,))
            success = cursor.rowcount > 0
            
            conn.commit()
            logger.info(f"Удален контент: ID {content_id}")
            return success
        except Exception as e:
            logger.error(f"Ошибка при удалении контента: {e}")
            return False
        finally:
            conn.close()

    def diagnose_database(self) -> bool:
        """Диагностика базы данных"""
        print("=== ДИАГНОСТИКА БАЗЫ ДАННЫХ ===")
        
        # Проверяем файл базы данных
        if not os.path.exists(self.db_name):
            print(f"❌ Файл базы данных '{self.db_name}' не существует")
            return False
        
        print(f"✅ Файл базы данных '{self.db_name}' существует")
        
        # Проверяем подключение и таблицы
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Проверяем основные таблицы
            required_tables = ['users', 'subjects', 'teachers', 'lectures', 'practices', 'useful_content']
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [table[0] for table in cursor.fetchall()]
            
            print("Существующие таблицы:", existing_tables)
            
            for table in required_tables:
                if table in existing_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"✅ Таблица '{table}': {count} записей")
                else:
                    print(f"❌ Таблица '{table}' отсутствует")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при диагностике: {e}")
            return False
    

    def get_all_schedule(self) -> List[Dict[str, Any]]:
        """Получение всех расписаний"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schedule_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('SELECT * FROM schedule_files ORDER BY uploaded_date DESC')
            schedules = [dict(row) for row in cursor.fetchall()]
            return schedules
        except Exception as e:
            logger.error(f"Ошибка при получении расписаний: {e}")
            return []
        finally:
            conn.close()

    def add_schedule(self, title: str, file_path: str) -> Optional[int]:
        """Добавление расписания"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO schedule_files (title, file_path) VALUES (?, ?)',
                (title, file_path)
            )
            schedule_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Добавлено расписание: {title}, ID: {schedule_id}")
            return schedule_id
        except Exception as e:
            logger.error(f"Ошибка при добавлении расписания: {e}")
            return None
        finally:
            conn.close()

    def get_schedule(self, schedule_id: int) -> Optional[Dict[str, Any]]:
        """Получение расписания по ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM schedule_files WHERE id = ?', (schedule_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Ошибка при получении расписания: {e}")
            return None
        finally:
            conn.close()

    def delete_schedule(self, schedule_id: int) -> bool:
        """Удаление расписания"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM schedule_files WHERE id = ?", (schedule_id,))
            success = cursor.rowcount > 0
            conn.commit()
            logger.info(f"Удалено расписание: ID {schedule_id}")
            return success
        except Exception as e:
            logger.error(f"Ошибка при удалении расписания: {e}")
            return False
        finally:
            conn.close()

class NotificationType(Enum):
    TECH_BREAK = "Технический перерыв"
    MEETING = "Совещание"
    ANNOUNCEMENT = "Объявление"
    OTHER = "Другое"

class NotificationManager:
    def __init__(self):
        self.active_notifications: Dict[int, Dict] = {}
        self.notification_id_counter = 1
    
    async def send_notification_to_all(self, application, message: str, notification_type: NotificationType, delay_minutes: int = 0):
        """Отправляет уведомление всем пользователям"""
        notification_id = self.notification_id_counter
        self.notification_id_counter += 1
        
        notification_data = {
            'id': notification_id,
            'type': notification_type,
            'message': message,
            'timestamp': datetime.now(),
            'scheduled_time': datetime.now() + timedelta(minutes=delay_minutes) if delay_minutes > 0 else None
        }
        
        self.active_notifications[notification_id] = notification_data
        
        # Форматируем сообщение
        formatted_message = self._format_notification(message, notification_type, delay_minutes)
        
        # Отправляем немедленно или с задержкой
        if delay_minutes > 0:
            asyncio.create_task(
                self._send_delayed_notification(application, notification_id, formatted_message, delay_minutes)
            )
        else:
            await self._broadcast_message(application, formatted_message)
        
        return notification_id
    
    def _format_notification(self, message: str, notification_type: NotificationType, delay_minutes: int) -> str:
        """Форматирует уведомление"""
        emoji = {
            NotificationType.TECH_BREAK: "🔧",
            NotificationType.MEETING: "📅",
            NotificationType.ANNOUNCEMENT: "📢",
            NotificationType.OTHER: "ℹ️"
        }
        
        time_info = ""
        if delay_minutes > 0:
            time_info = f"\n⏰ Через {delay_minutes} минут"
        
        return (
            f"{emoji[notification_type]} **{notification_type.value}**\n"
            f"{message}{time_info}\n"
            f"_{datetime.now().strftime('%H:%M')}_"
        )
    
    async def _send_delayed_notification(self, application, notification_id: int, message: str, delay_minutes: int):
        """Отправляет отложенное уведомление"""
        await asyncio.sleep(delay_minutes * 60)
        
        if notification_id in self.active_notifications:
            await self._broadcast_message(application, message)
            # Удаляем из активных после отправки
            del self.active_notifications[notification_id]
    
    async def _broadcast_message(self, application, message: str):
        """Отправляет сообщение всем пользователям"""
        # Здесь нужно добавить логику получения списка всех пользователей
        # Пока что просто логируем
        logger.info(f"Broadcasting message to all users: {message}")
        
        # В реальном приложении здесь должен быть код для отправки всем пользователям бота
        # Например: 
        # for user_id in all_user_ids:
        #     try:
        #         await application.bot.send_message(chat_id=user_id, text=message, parse_mode='Markdown')
        #     except Exception as e:
        #         logger.error(f"Failed to send message to {user_id}: {e}")
        
        print(f"Уведомление для всех: {message}")
class TeamManager:
    def __init__(self):
        self.teams: Dict[str, List[str]] = {}
        self.user_teams: Dict[str, str] = {}
        self.notification_manager = NotificationManager()  # Добавить эту строку
    
    async def admin_send_notification(self, application, message: str, notification_type: NotificationType, delay_minutes: int = 0):
        """Метод для администратора для отправки уведомлений"""
        return await self.notification_manager.send_notification_to_all(
            application, message, notification_type, delay_minutes
        )
# =============================================================================
# ОБНОВЛЕННЫЙ КЛАСС LectureBot С НОВЫМИ ФУНКЦИЯМИ
# =============================================================================

class EnhancedLectureBot:
    """Улучшенный бот с системой управления кодом и самообучением"""
    
    def __init__(self):
        self.db = Database()
        self.application = None
        self.ai_assistant = None
        self.code_manager = BotCodeManager(self)
        self.start_time = datetime.now()
        self._initialize_bot()  
        self.file_manager = FileManager()
        self.mass_upload_handler = MassUploadHandler(self.db, self.file_manager)
        # Текст для помощника
        self.helper_text = "👋 Привет! Помогаю в разработке курсовых (от 2000), а также в подготовке отчетов учебных и производственных практик (от 500), проектных работ и докладов (от 200), практических заданий и конспектов (от 35). Создаю сайты (html, css, js, react, vue, django, php, nodeJS, tilda) и пишу программы (c#, pascal, python, delphia)"
        self.helper_contact = "@RaffLik"
        
    def _initialize_bot(self):
        """Инициализация улучшенного бота"""
        try:
            # Инициализация приложения Telegram с увеличенными лимитами
            builder = Application.builder().token(BOT_TOKEN)
            
            # Увеличиваем лимиты для загрузки файлов до 2GB
            builder = builder.pool_timeout(120) \
                        .connect_timeout(120) \
                        .read_timeout(120) \
                        .write_timeout(120)
            
            self.application = builder.build()
            
            # Инициализация улучшенного ИИ-помощника
            ai_config = AIConfig(
                api_key=HUGGINGFACE_API_KEY,
                models=HUGGING_FACE_MODELS,
                default_model_index=0
            )
            self.ai_assistant = EnhancedAIAssistant(ai_config)
            
            self.setup_handlers()
            logger.info("✅ Улучшенный бот успешно инициализирован")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации улучшенного бота: {e}")
            raise
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик нажатий на кнопки с исправленными обработчиками"""
        query = update.callback_query
        await query.answer()
        data = query.data
        
        try:
            logger.info(f"Обработка callback_data: {data}")
            
            # Обработка основных кнопок меню
            if data == "schedule":
                await self.show_schedule(query, context)
            elif data == "helper":
                await self.show_helper(query, context)
            elif data == "useful_info":
                await self.show_useful_info_safe(query, context)
            elif data == "support":
                await self.show_support(query, context)
            elif data == "donate":
                await self.show_donate_callback(query, context)
            elif data == "ai_assistant":
                await self.show_ai_chat(query, context)
            elif data == "subjects":
                await self.show_subjects(query, context)
            
            # Админ-панель и управление
            elif data == "admin_panel":
                await self.show_admin_panel(query, context)
            elif data == "code_manager":
                await self.code_manager_panel_callback(query, context)
            elif data == "back_to_menu":
                await self.show_main_menu(Update(update_id=0, callback_query=query), context)
            
            # Управление кодом
            elif data == "system_status":
                await self.show_system_status(query, context)
            elif data == "update_code":
                await self.update_code_from_github_callback(query, context)
            elif data == "cleanup_temp":
                await self.cleanup_temp_files_callback(query, context)
            elif data == "restart_bot":
                await self.restart_bot_confirmation(query, context)
            elif data == "view_files":
                await self.start_file_view(query, context)
            elif data == "execute_command":
                await self.start_command_execution(query, context)
            elif data == "create_backup":
                await self.create_system_backup(query, context)
            elif data == "list_backups":
                await self.list_system_backups(query, context)
            elif data == "force_learning":
                await self.force_learning_callback(query, context)
            elif data == "confirm_restart":
                await self.confirm_restart(query, context)
            
            # ИИ и обучение
            elif data == "ai_stats":
                await self.show_ai_stats_callback(query, context)
            elif data == "train_dataset":
                await self.show_dataset_training(query, context)
            elif data == "upload_dataset":
                await self.start_upload_dataset(query, context)
            elif data == "upload_github_dataset":
                await self.start_upload_github_dataset(query, context)
            elif data == "manage_datasets":
                await self.show_manage_datasets(query, context)
            elif data == "ai_clear_history":
                await self.clear_ai_history(query, context)
            elif data == "diagnose_training":
                await self.diagnose_training(query, context)
            
            # Расписание
            elif data == "manage_schedule":
                await self.manage_schedule(query, context)
            elif data == "upload_schedule":
                await self.start_upload_schedule(query, context)
            elif data == "view_schedule":
                await self.show_schedule_list(query, context)
            
            # Полезная информация
            elif data == "manage_useful_info":
                await self.manage_useful_info_safe(query, context)
            elif data == "upload_useful_info":
                await self.start_upload_useful_info_safe(query, context)
            elif data == "view_useful_info":
                await self.show_useful_info_list_safe(query, context)
            
            # Предметы и преподаватели
            elif data == "add_subject":
                await self.start_add_subject(query, context)
            elif data == "add_teacher":
                await self.start_add_teacher(query, context)
            
            # Загрузка файлов - ДОБАВЛЯЕМ ОБРАБОТЧИКИ
            elif data == "upload_file":
                await self.start_single_upload(query, context)
            elif data == "upload_file":
                await self.start_single_upload(query, context)
            elif query.data == "mass_upload":
                mass_upload_handler = MassUploadHandler(self.db, self.file_manager)
                await mass_upload_handler.start_mass_upload_simple(query, context)
            elif data == "delete_files":
                await self.show_delete_files_menu(query, context)
            # Обработчики массовой загрузки
            elif data.startswith("mass_upload_subject_"):
                await self.select_subject_mass_upload(query, context)
            elif data.startswith("mass_upload_type_"):
                await self.select_file_type_mass_upload(query, context)
            elif data == "mass_upload_finish":
                await self.finish_mass_upload(query, context)
            elif data == "mass_upload_confirm":
                await self.confirm_mass_upload(query, context)
            elif data.startswith("mass_upload_back_"):
                await self.navigate_back_mass_upload(query, context)
            elif data == "cancel_mass_upload":
                await self.cancel_mass_upload(query, context)
                # Обработчики удаления файлов
            elif data == "delete_lectures_menu":
                await self.delete_lectures_menu(query, context)
            elif data == "delete_practices_menu":
                await self.delete_practices_menu(query, context)
            elif data == "delete_schedules_menu":
                await self.delete_schedules_menu(query, context)
            elif data == "delete_useful_menu":
                await self.delete_useful_menu(query, context)
            elif data.startswith("delete_lectures_subject_"):
                subject_id = int(data.split("_")[-1])
                await self.show_lectures_for_deletion(query, subject_id, context)
            elif data.startswith("delete_practices_subject_"):
                subject_id = int(data.split("_")[-1])
                await self.show_practices_for_deletion(query, subject_id, context)
            elif data.startswith("delete_subject_"):
                await self.show_subject_files_for_deletion(query, context)
            elif data.startswith("delete_file_"):
                await self.confirm_file_deletion(query, context)
            elif data.startswith("confirm_delete_"):
                await self.execute_file_deletion(query, context)
            elif data.startswith("cancel_delete_"):
                await self.cancel_file_deletion(query, context)
            elif data == "delete_back_subjects":
                await self.show_delete_files_menu(query, context)
            elif data == "delete_back_files":
                subject_id = int(data.split('_')[-1]) if data.split('_')[-1].isdigit() else None
                if subject_id:
                    await self.show_subject_files_for_deletion(query, context, subject_id)
            
            # Логи
            elif data == "view_logs":
                await self.show_logs(query, context)
            
            # Обработка динамических callback данных
            elif data.startswith("subject_"):
                subject_id = int(data.split("_")[1])
                await self.show_subject_content(query, subject_id, context)
            elif data.startswith("lecture_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    subject_id = int(parts[1])
                    lecture_num = int(parts[2])
                    await self.send_lecture(query, subject_id, lecture_num, context)
            elif data.startswith("practice_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    subject_id = int(parts[1])
                    practice_num = int(parts[2])
                    await self.send_practice(query, subject_id, practice_num, context)
            elif data.startswith("show_lectures_"):
                subject_id = int(data.split("_")[2])
                await self.show_lectures_list(query, subject_id, context)
            elif data.startswith("show_practices_"):
                subject_id = int(data.split("_")[2])
                await self.show_practices_list(query, subject_id, context)
            elif data.startswith("download_schedule_"):
                schedule_id = int(data.split("_")[2])
                await self.send_schedule_file(query, schedule_id, context)
            elif data.startswith("delete_schedule_"):
                schedule_id = int(data.split("_")[2])
                await self.delete_schedule(query, schedule_id, context)
            elif data.startswith("download_useful_"):
                content_id = int(data.split("_")[2])
                await self.send_useful_file(query, content_id, context)
            elif data.startswith("delete_useful_"):
                content_id = int(data.split("_")[2])
                await self.delete_useful_content(query, content_id, context)
            elif data.startswith("train_on_dataset_"):
                dataset_name = data.split("_", 3)[-1]
                await self.start_dataset_training(query, context, dataset_name)
            elif data.startswith("delete_dataset_"):
                dataset_name = data.split("_", 2)[-1]
                await self.delete_dataset(query, context, dataset_name)
            elif data.startswith("select_subject_"):
                subject_id = int(data.split("_")[2])
                await self.handle_select_subject_for_teacher(query, subject_id, context)
            elif data.startswith("upload_subject_"):
                subject_id = int(data.split("_")[2])
                await self.handle_select_upload_subject(query, subject_id, context)
            elif data.startswith("upload_type_"):
                upload_type = data.split("_")[2]
                await self.handle_select_upload_type(query, upload_type, context)
            elif data.startswith("view_logs_"):
                log_type = data.split("_")[2]
                await self.show_logs_by_type(query, context, log_type)
            elif data == "view_all_datasets":
                await self.view_all_datasets(query, context)
            else:
                logger.warning(f"Неизвестный callback_data: {data}")
                await query.answer("❌ Команда не распознана", show_alert=True)
                
        except (ValueError, IndexError) as e:
            logger.error(f"Ошибка обработки callback_data '{data}': {e}")
            await query.answer("❌ Ошибка обработки команды", show_alert=True)
        except Exception as e:
            logger.error(f"Неожиданная ошибка в button_handler: {e}")
            await query.answer("❌ Произошла ошибка", show_alert=True)

    async def show_ai_chat(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать чат с ИИ-помощником"""
        if 'ai_conversation' not in context.user_data:
            context.user_data['ai_conversation'] = []

        stats = self.ai_assistant.get_stats()

        welcome_text = (
            "🤖 ИИ-помощник (EnhancedSelfLearningAI)\n\n"
            "Привет! Я ваш AI-ассистент с самообучением. Я могу помочь вам с:\n"
            "• Объяснением сложных тем\n"
            "• Ответами на учебные вопросы\n"
            "• Решением задач\n"
            "• Подготовкой к экзаменам\n\n"
            "Просто напишите ваш вопрос, и я постараюсь помочь!\n\n"
            "💡 Модель постоянно обучается и улучшает свои ответы."
        )

        keyboard = [
            [InlineKeyboardButton("🧹 Очистить историю", callback_data="ai_clear_history")],
            [InlineKeyboardButton("📊 Статистика ИИ", callback_data="ai_stats")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]

        if query.from_user.id in ADMIN_IDS:
            keyboard.insert(0, [InlineKeyboardButton("📚 Обучить на датасете", callback_data="train_dataset")])

        await self.edit_message_with_cleanup(
            query, context,
            welcome_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        context.user_data['state'] = 'ai_chat'

    async def show_lectures_list(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Показать список лекций для предмета"""
        logger.info(f"Показать список лекций для subject_id: {subject_id}")
        
        lectures = self.db.get_lectures(subject_id)
        subject = self.db.get_subject(subject_id)
        
        if not subject:
            await self.edit_message_with_cleanup(query, context, "❌ Предмет не найден")
            return
            
        logger.info(f"Найдено лекций: {len(lectures)}")
        
        if not lectures:
            await self.edit_message_with_cleanup(
                query, context,
                f"📓 {subject['name']}\n\nПока нет доступных лекций.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data=f"subject_{subject_id}")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
                ])
            )
            return
        
        keyboard = []
        for lecture in sorted(lectures, key=lambda x: x['number']):
            if lecture.get('name'):
                button_text = f"📓 {lecture['name']}"
            else:
                button_text = f"📓 Лекция #{lecture['number']}"
            
            logger.info(f"Добавляем лекцию: {button_text}")
            keyboard.append([
                InlineKeyboardButton(
                    button_text, 
                    callback_data=f"lecture_{subject_id}_{lecture['number']}"
                )
            ])

        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data=f"subject_{subject_id}")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")])
        
        await self.edit_message_with_cleanup(
            query, context,
            f"📓 {subject['name']}\nВыберите лекцию:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def show_schedule(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать расписание"""
        schedules = self.db.get_all_schedule()
        
        if not schedules:
            await self.edit_message_with_cleanup(
                query, context,
                "📅 Расписание\n\nПока нет доступного расписания.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
                ])
            )
            return
        
        keyboard = []
        for schedule in schedules:
            keyboard.append([
                InlineKeyboardButton(
                    f"📅 {schedule['title']}", 
                    callback_data=f"download_schedule_{schedule['id']}"
                )
            ])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        
        await self.edit_message_with_cleanup(
            query, context,
            "📅 Расписание\nВыберите файл для скачивания:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def send_lecture(self, query, subject_id: int, lecture_num: int, context: ContextTypes.DEFAULT_TYPE):
        """Отправить лекцию"""
        logger.info(f"Попытка отправки лекции: subject_id={subject_id}, lecture_num={lecture_num}")
        try:
            lecture = self.db.get_lecture(subject_id, lecture_num)
            subject = self.db.get_subject(subject_id)
            
            if not lecture or not subject:
                await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Лекция не найдена в базе данных")
                await self.show_lectures_list(query, subject_id, context)
                return
            
            file_path = lecture['file_path']
            
            if not os.path.exists(file_path):
                await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Файл не найден на сервере")
                await self.show_lectures_list(query, subject_id, context)
                return
            
            lecture_name = ""
            if lecture.get('name'):
                lecture_name = f"\n📝 {lecture['name']}"
            else:
                lecture_name = f"\n🔢 Номер: #{lecture_num}"
            
            caption = f"📓 {subject['name']}{lecture_name}"
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            with open(file_path, 'rb') as file:
                if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    await query.message.reply_photo(
                        photo=file,
                        caption=caption
                    )
                elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    await query.message.reply_video(
                        video=file,
                        caption=caption
                    )
                else:
                    await query.message.reply_document(
                        document=file,
                        caption=caption
                    )
            
            await self.send_message_with_cleanup(
                Update(update_id=0, callback_query=query), context,
                "Файл отправлен!",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад к лекциям", callback_data=f"show_lectures_{subject_id}")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
                ])
            )
            
        except Exception as e:
            logger.error(f"Ошибка при отправке лекции: {e}")
            await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Ошибка при отправке файла")
            await self.show_lectures_list(query, subject_id, context)

    
    async def handle_file_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Обработка файловых сообщений"""
            user_state = context.user_data.get('state', '')
            
            if user_state == 'uploading_dataset':
                await self.save_dataset_file(update, context)
            elif user_state == 'uploading_schedule':
                await self.save_schedule_file(update, context)
            elif user_state == 'uploading_useful_info':
                await self.save_useful_info_file(update, context)
            elif user_state == 'single_upload_file':
                await self.save_single_file(update, context)
            else:
                await self.send_message_with_cleanup(update, context, "Пожалуйста, сначала начните процесс загрузки файла через админ-панель.")

    async def show_practices_list(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Показать список практических работ для предмета"""
        logger.info(f"Показать список практических для subject_id: {subject_id}")
        
        practices = self.db.get_practices(subject_id)
        subject = self.db.get_subject(subject_id)
        
        if not subject:
            await self.edit_message_with_cleanup(query, context, "❌ Предмет не найден")
            return
            
        logger.info(f"Найдено практических работ: {len(practices)}")
        
        if not practices:
            await self.edit_message_with_cleanup(
                query, context,
                f"📝 {subject['name']}\n\nПока нет доступных практических работ.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data=f"subject_{subject_id}")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
                ])
            )
            return
        
        keyboard = []
        for practice in sorted(practices, key=lambda x: x['number']):
            if practice.get('name'):
                button_text = f"📝 {practice['name']}"
            else:
                button_text = f"📝 Практическая #{practice['number']}"
            
            logger.info(f"Добавляем практическую: {button_text}")
            keyboard.append([
                InlineKeyboardButton(
                    button_text, 
                    callback_data=f"practice_{subject_id}_{practice['number']}"
                )
            ])

        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data=f"subject_{subject_id}")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")])
        
        await self.edit_message_with_cleanup(
            query, context,
            f"📝 {subject['name']}\nВыберите практическую работу:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def send_practice(self, query, subject_id: int, practice_num: int, context: ContextTypes.DEFAULT_TYPE):
        """Отправить практическую работу"""
        logger.info(f"Попытка отправки практической: subject_id={subject_id}, practice_num={practice_num}")
        try:
            practice = self.db.get_practice(subject_id, practice_num)
            subject = self.db.get_subject(subject_id)
            
            if not practice or not subject:
                await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Практическая работа не найдена в базе данных")
                await self.show_practices_list(query, subject_id, context)
                return
            
            file_path = practice['file_path']
            
            if not os.path.exists(file_path):
                await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Файл не найден на сервере")
                await self.show_practices_list(query, subject_id, context)
                return
            
            practice_name = ""
            if practice.get('name'):
                practice_name = f"\n📝 {practice['name']}"
            else:
                practice_name = f"\n🔢 Номер: #{practice_num}"
            
            caption = f"📝 {subject['name']}{practice_name}"
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            with open(file_path, 'rb') as file:
                if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    await query.message.reply_photo(
                        photo=file,
                        caption=caption
                    )
                elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    await query.message.reply_video(
                        video=file,
                        caption=caption
                    )
                else:
                    await query.message.reply_document(
                        document=file,
                        caption=caption
                    )
            
            await self.send_message_with_cleanup(
                Update(update_id=0, callback_query=query), context,
                "Файл отправлен!",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад к практическим", callback_data=f"show_practices_{subject_id}")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
                ])
            )
            
        except Exception as e:
            logger.error(f"Ошибка при отправке практической работы: {e}")
            await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Ошибка при отправке файла")
            await self.show_practices_list(query, subject_id, context)


    async def show_subject_content(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Показать контент предмета (лекции и практические)"""
        logger.info(f"Показать контент предмета: {subject_id}")
        
        subject = self.db.get_subject(subject_id)
        if not subject:
            await self.edit_message_with_cleanup(query, context, "❌ Предмет не найден")
            return
            
        lectures = self.db.get_lectures(subject_id)
        practices = self.db.get_practices(subject_id)
        
        logger.info(f"Лекций: {len(lectures)}, Практических: {len(practices)}")
        
        keyboard = []
        
        if lectures:
            keyboard.append([InlineKeyboardButton("📓 Лекции", callback_data=f"show_lectures_{subject_id}")])
        
        if practices:
            keyboard.append([InlineKeyboardButton("📝 Практические работы", callback_data=f"show_practices_{subject_id}")])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад к предметам", callback_data="subjects")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")])
        
        text = f"📖 {subject['name']}\n\n"
        text += f"📓 Лекций: {len(lectures)}\n"
        text += f"📝 Практических работ: {len(practices)}"
        
        await self.edit_message_with_cleanup(
            query, context,
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def show_subjects(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать список предметов"""
        subjects = self.db.get_all_subjects()
        
        if not subjects:
            await self.edit_message_with_cleanup(
                query, context,
                "📚 Предметы\n\nПока нет доступных предметов.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
                ])
            )
            return
        
        keyboard = []
        for subject in subjects:
            keyboard.append([
                InlineKeyboardButton(
                    f"📖 {subject['name']}", 
                    callback_data=f"subject_{subject['id']}"
                )
            ])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        
        await self.edit_message_with_cleanup(
            query, context,
            "📚 Выберите предмет:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений с новыми состояниями"""
        user_state = context.user_data.get('state', '')
        user_message = update.message.text.strip()
        
        #logger.info(f"Обр" + "Показать меню удаления файлов" + "аботка текстового сообщения: состояние={user_state}, сообщение={user_message}")
        
        try:
            # Обработка состояний
            if user_state == 'ai_chat':
                await self.handle_ai_message(update, context)
            elif user_state == 'viewing_file':
                await self.handle_file_viewing(update, context, user_message)
            elif user_state == 'executing_command':
                await self.handle_command_execution(update, context, user_message)
            elif user_state == 'adding_subject':
                await self.handle_add_subject(update, context, user_message)
            elif user_state == 'adding_teacher_name':
                await self.handle_add_teacher(update, context, user_message)
            elif user_state == 'single_upload_number':
                await self.handle_upload_number(update, context, user_message)
            elif user_state == 'uploading_github_dataset':
                await self.handle_github_url(update, context, user_message)
            elif user_state == 'uploading_schedule':
                await self.handle_schedule_upload(update, context, user_message)
            elif user_state == 'uploading_useful_info':
                await self.handle_useful_info_upload(update, context, user_message)
            # Добавляем обработку массовой загрузки
            elif user_state == 'mass_upload':
                user_id = update.message.from_user.id
                
                if user_id not in self.temp_uploads:
                    await update.message.reply_text(
                        "❌ Сейчас не активна массовая загрузка. "
                        "Используйте /mass_upload для начала загрузки."
                    )
                    return ConversationHandler.END
                
                # Если пользователь в процессе массовой загрузки, напоминаем о файлах
                await update.message.reply_text(
                    "📎 Пожалуйста, отправляйте файлы для загрузки.\n"
                    "Когда закончите, нажмите '✅ Завершить загрузку'\n\n"
                    "❌ Для отмены используйте /cancel"
                )
                return UPLOAD_FILES
            # Если состояние не определено, показываем главное меню
            else:
                await self.show_main_menu(update, context)
                
        except Exception as e:
            logger.error(f"Ошибка в handle_text_message: {e}")
            await self.send_message_with_cleanup(
                update, context,
                "❌ Произошла ошибка при обработке сообщения. Попробуйте еще раз.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
                ])
            )

    def run_bot(self):
        """Запуск бота"""
        logger.info("Бот запускается")
        try:
            self.application.run_polling(
                drop_pending_updates=True,
                timeout=60,
            )
        except Exception as e:
            logger.error(f"Ошибка в run_polling: {e}")
            raise

    async def shutdown(self):
        """Закрытие ресурсов при завершении работы"""
        pass

    async def start_add_subject(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать добавление предмета"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        context.user_data.clear()
        context.user_data['state'] = 'adding_subject'
        
        await self.edit_message_with_cleanup(
            query, context,
            "➕ Добавление нового предмета\n\n"
            "Введите название предмета:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")]
            ])
        )

    async def start_add_teacher(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать добавление преподавателя"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        # Получаем список предметов для выбора
        subjects = self.db.get_all_subjects()
        
        if not subjects:
            await self.edit_message_with_cleanup(
                query, context,
                "👨‍🏫 Добавление преподавателя\n\n"
                "❌ Сначала добавьте предметы, чтобы привязать преподавателя.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Добавить предмет", callback_data="add_subject")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
            return
        
        context.user_data.clear()
        context.user_data['state'] = 'adding_teacher_subject'
        
        # Создаем клавиатуру с предметами
        keyboard = []
        for subject in subjects:
            keyboard.append([InlineKeyboardButton(
                f"📖 {subject['name']}", 
                callback_data=f"select_subject_{subject['id']}"
            )])
        
        keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")])
        
        await self.edit_message_with_cleanup(
            query, context,
            "👨‍🏫 Добавление преподавателя\n\n"
            "Выберите предмет для преподавателя:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


    async def show_lectures_for_deletion(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Показать лекции для удаления"""
        lectures = self.db.get_lectures(subject_id)
        subject = self.db.get_subject(subject_id)
        
        if not lectures:
            await self.edit_message_with_cleanup(
                query, context,
                f"❌ В предмете '{subject['name']}' нет лекций для удаления",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="delete_lectures_menu")]
                ])
            )
            return
        
        keyboard = []
        for lecture in sorted(lectures, key=lambda x: x['number']):
            keyboard.append([
                InlineKeyboardButton(
                    f"🗑️ Лекция #{lecture['number']}", 
                    callback_data=f"confirm_delete_lecture_{lecture['id']}"
                )
            ])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="delete_lectures_menu")])
        
        await self.edit_message_with_cleanup(
            query, context,
            f"🗑️ Удаление лекций: {subject['name']}\n\nВыберите лекцию для удаления:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def show_practices_for_deletion(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Показать практические работы для удаления"""
        practices = self.db.get_practices(subject_id)
        subject = self.db.get_subject(subject_id)
        
        if not practices:
            await self.edit_message_with_cleanup(
                query, context,
                f"❌ В предмете '{subject['name']}' нет практических работ для удаления",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="delete_practices_menu")]
                ])
            )
            return
        
        keyboard = []
        for practice in sorted(practices, key=lambda x: x['number']):
            keyboard.append([
                InlineKeyboardButton(
                    f"🗑️ Практическая #{practice['number']}", 
                    callback_data=f"confirm_delete_practice_{practice['id']}"
                )
            ])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="delete_practices_menu")])
        
        await self.edit_message_with_cleanup(
            query, context,
            f"🗑️ Удаление практических работ: {subject['name']}\n\nВыберите работу для удаления:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def confirm_delete_lecture(self, query, lecture_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Подтверждение удаления лекции"""
        # Здесь должна быть логика получения информации о лекции
        # и подтверждения удаления
        pass

    async def confirm_delete_practice(self, query, practice_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Подтверждение удаления практической работы"""
        # Здесь должна быть логика получения информации о практике
        # и подтверждения удаления
        pass

    async def start_file_view(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать просмотр файлов"""
        context.user_data['state'] = 'viewing_file'
        
        await self.edit_message_with_cleanup(
            query, context,
            "📁 Просмотр файлов\n\n"
            "Введите путь к файлу для просмотра:\n\n"
            "Примеры:\n"
            "• bot.py\n"
            "• bot_database.db (только информация)\n"
            "• training_datasets/data.json\n\n"
            "❌ Для отмены используйте /cancel",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отмена", callback_data="code_manager")]
            ])
        )

    async def restart_bot_confirmation(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Подтверждение перезапуска бота"""
        await self.edit_message_with_cleanup(
            query, context,
            "⚠️ Подтверждение перезапуска\n\n"
            "Вы уверены, что хотите перезапустить бота?\n\n"
            "Бот будет временно недоступен на несколько секунд.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Да, перезапустить", callback_data="confirm_restart")],
                [InlineKeyboardButton("❌ Отмена", callback_data="code_manager")]
            ])
        )

    async def show_admin_panel(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать админ-панель из команды"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
            
        keyboard = [
            [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
            [InlineKeyboardButton("📤 Одиночная загрузка", callback_data="upload_file")],
            [InlineKeyboardButton("📚 Массовая загрузка", callback_data="mass_upload")],  # Оставляем, но теперь есть обработчик
            [InlineKeyboardButton("🗑️ Удаление файлов", callback_data="delete_files")],
            [InlineKeyboardButton("➕ Добавить предмет", callback_data="add_subject")],
            [InlineKeyboardButton("👨‍🏫 Добавить преподавателя", callback_data="add_teacher")],
            [InlineKeyboardButton("📅 Управление расписанием", callback_data="manage_schedule")],
            [InlineKeyboardButton("📦 Управление полезной инфо", callback_data="manage_useful_info")],
            [InlineKeyboardButton("📋 Просмотр логов", callback_data="view_logs")],
            [InlineKeyboardButton("🤖 Статистика ИИ", callback_data="ai_stats")],
            [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")],  # Новая кнопка
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
        ]
        
        await self.edit_message_with_cleanup(
            query, context,
            "⚙️ Админ-панель\nВыберите действие:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def list_backups(self) -> List[Dict[str, Any]]:
        """Получить список бэкапов"""
        backups = []
        try:
            for item in os.listdir(self.backup_dir):
                backup_path = os.path.join(self.backup_dir, item)
                if os.path.isdir(backup_path):
                    info_file = os.path.join(backup_path, "backup_info.json")
                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        backups.append({
                            "name": item,
                            "timestamp": info.get("timestamp", ""),
                            "size": info.get("total_size", "0 MB")
                        })
        except Exception as e:
            logger.error(f"Ошибка получения списка бэкапов: {e}")
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

    # Добавляем остальные методы, которые были в оригинальном классе
    async def create_backup(self) -> Tuple[bool, str]:
        """Асинхронная версия создания бэкапа"""
        return self.create_backup_sync()

    def view_file(self, file_path: str) -> Tuple[bool, str, str]:
        """Просмотреть содержимое файла"""
        try:
            if not os.path.exists(file_path):
                return False, "", f"❌ Файл не найден: {file_path}"
            
            # Проверяем размер файла
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024:  # 100KB лимит
                return False, "", f"❌ Файл слишком большой ({file_size} байт). Максимум 100KB."
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return True, content, f"✅ Файл прочитан: {file_path}"
            
        except Exception as e:
            logger.error(f"Ошибка чтения файла: {e}")
            return False, "", f"❌ Ошибка чтения файла: {str(e)}"

    async def edit_file(self, file_path: str, new_content: str) -> Tuple[bool, str]:
        """Редактировать файл"""
        try:
            if not os.path.exists(file_path):
                return False, f"❌ Файл не найден: {file_path}"
            
            # Создаем бэкап файла
            backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)
            
            # Записываем новый контент
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True, f"✅ Файл успешно обновлен. Бэкап: {backup_path}"
            
        except Exception as e:
            logger.error(f"Ошибка редактирования файла: {e}")
            return False, f"❌ Ошибка редактирования: {str(e)}"

    async def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """Выполнить команду на сервере"""
        try:
            # Безопасность: ограничиваем опасные команды
            dangerous_commands = ['rm -rf', 'format', 'dd', 'mkfs', 'chmod 777']
            if any(cmd in command for cmd in dangerous_commands):
                return False, "", "❌ Опасная команда запрещена"
            
            # Специальная команда для очистки временных файлов
            if command.strip() == "cleanup":
                success, message = await self.cleanup_temp_files()
                return success, message, "Команда очистки выполнена"
            
            # Выполняем команду
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            output = result.stdout if result.stdout else result.stderr
            success = result.returncode == 0
            
            return success, output, f"Код возврата: {result.returncode}"
            
        except subprocess.TimeoutExpired:
            return False, "", "❌ Команда превысила лимит времени (30 секунд)"
        except Exception as e:
            logger.error(f"Ошибка выполнения команды: {e}")
            return False, "", f"❌ Ошибка выполнения: {str(e)}"
        
    async def confirm_restart(self, query, context):
        """Подтверждение перезапуска"""
        success, message = self.code_manager.restart_bot()
        await self.edit_message_with_cleanup(query, context, message)

    async def view_all_datasets(self, query, context):
        """Показать все датасеты"""
        datasets = self.ai_assistant.get_datasets_info()
        
        if not datasets:
            await self.edit_message_with_cleanup(
                query, context,
                "📚 Все датасеты\n\nНет доступных датасетов.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📤 Загрузить датасет", callback_data="upload_dataset")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="manage_datasets")]
                ])
            )
            return
        
        message_text = "📚 Все датасеты:\n\n"
        for i, dataset in enumerate(datasets, 1):
            message_text += f"{i}. {dataset['filename']} ({dataset['size_mb']} MB)\n"
        
        keyboard = []
        for dataset in datasets:
            filename = dataset['filename']
            display_name = filename[:15] + "..." if len(filename) > 15 else filename
            
            keyboard.append([
                InlineKeyboardButton(f"🎯 {display_name}", callback_data=f"train_on_dataset_{filename}"),
                InlineKeyboardButton(f"🗑️", callback_data=f"delete_dataset_{filename}")
            ])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="manage_datasets")])
        
        await self.edit_message_with_cleanup(
            query, context,
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def start_command_execution(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать выполнение команды"""
        context.user_data['state'] = 'executing_command'
        
        await self.edit_message_with_cleanup(
            query, context,
            "⚙️ Выполнение команды\n\n"
            "Введите команду для выполнения на сервере:\n\n"
            "Примеры безопасных команд:\n"
            "• ls -la\n"
            "• pwd\n"
            "• python --version\n"
            "• pip list\n\n"
            "⚠️ Опасные команды заблокированы\n"
            "❌ Для отмены используйте /cancel",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отмена", callback_data="code_manager")]
            ])
        )

    async def create_system_backup(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Создать бэкап системы"""
        await self.edit_message_with_cleanup(
            query, context,
            "💾 Создание бэкапа системы...\n\n"
            "Пожалуйста, подождите."
        )
        
        # Исправленный вызов - теперь это асинхронный метод
        success, message = await self.code_manager.create_backup()
        
        await self.edit_message_with_cleanup(
            query, context,
            message,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📋 Список бэкапов", callback_data="list_backups")],
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")]
            ])
        )

    async def list_system_backups(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать список бэкапов"""
        backups = await self.code_manager.list_backups()
        
        if not backups:
            await self.edit_message_with_cleanup(
                query, context,
                "📋 Список бэкапов\n\n"
                "Бэкапы не найдены.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("💾 Создать бэкап", callback_data="create_backup")],
                    [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")]
                ])
            )
            return
        
        backup_text = "📋 Список бэкапов\n\n"
        for i, backup in enumerate(backups[:10]):  # Показываем первые 10
            backup_text += f"{i+1}. {backup['name']}\n"
            backup_text += f"   📅 {backup['timestamp'][:10]} | 💾 {backup['size']}\n\n"
        
        if len(backups) > 10:
            backup_text += f"... и еще {len(backups) - 10} бэкапов\n"
        
        await self.edit_message_with_cleanup(
            query, context,
            backup_text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💾 Создать бэкап", callback_data="create_backup")],
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")]
            ])
        )

    async def force_learning_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для принудительного обучения"""
        await self.force_learning(Update(update_id=0, callback_query=query), context)

    async def show_logs_by_type(self, query, context: ContextTypes.DEFAULT_TYPE, log_type: str):
        """Показать логи по типу с кликабельными кнопками"""
        # Добавляем проверку прав доступа
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        logs = self.db.get_logs(limit=50, level=log_type.upper() if log_type != 'all' else None)
        
        if not logs:
            await self.edit_message_with_cleanup(
                query, context,
                f"📋 Логи ({log_type})\n\nНет записей.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📋 Все логи", callback_data="view_logs_all")],
                    [InlineKeyboardButton("❌ Ошибки", callback_data="view_logs_error")],
                    [InlineKeyboardButton("⚠️ Предупреждения", callback_data="view_logs_warning")],
                    [InlineKeyboardButton("ℹ️ Инфо", callback_data="view_logs_info")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")]
                ])
            )
            return
        
        # Группируем логи по уровням для статистики
        error_count = len([log for log in logs if log['level'] == 'ERROR'])
        warning_count = len([log for log in logs if log['level'] == 'WARNING'])
        info_count = len([log for log in logs if log['level'] == 'INFO'])
        
        logs_text = (
            f"📋 Логи ({log_type})\n\n"
            f"❌ Ошибки: {error_count}\n"
            f"⚠️ Предупреждения: {warning_count}\n"
            f"ℹ️ Инфо: {info_count}\n\n"
            "Последние записи:\n"
        )
        
        # Показываем последние 8 записей
        for i, log in enumerate(logs[:8]):
            time_str = log['created_at'][11:19]  # Только время
            level_icon = "❌" if log['level'] == 'ERROR' else "⚠️" if log['level'] == 'WARNING' else "ℹ️"
            logs_text += f"\n{time_str} {level_icon} {log['message'][:60]}..."
        
        if len(logs) > 8:
            logs_text += f"\n\n... и еще {len(logs) - 8} записей"
        
        await self.edit_message_with_cleanup(
            query, context,
            logs_text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📋 Все логи", callback_data="view_logs_all")],
                [InlineKeyboardButton("❌ Ошибки", callback_data="view_logs_error")],
                [InlineKeyboardButton("⚠️ Предупреждения", callback_data="view_logs_warning")],
                [InlineKeyboardButton("ℹ️ Инфо", callback_data="view_logs_info")],
                [InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")]
            ])
        )

    async def update_code_from_github(self, repo_url: str = None, branch: str = "main") -> Tuple[bool, str]:
        """Асинхронная версия обновления кода из GitHub с улучшенной обработкой ошибок"""
        try:
            if not repo_url:
                repo_url = "https://github.com/Yaroslav858/bot.py.git"  
            
            temp_dir = "temp_update"
            
            # УЛУЧШЕННАЯ ОЧИСТКА: проверяем и полностью удаляем существующую директорию
            if os.path.exists(temp_dir):
                logger.info(f"Удаляем существующую директорию {temp_dir}")
                try:
                    # Рекурсивно удаляем всю директорию
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                    # Дополнительная проверка что директория удалена
                    if os.path.exists(temp_dir):
                        # Если не удалось удалить, пробуем другой подход
                        import subprocess
                        if os.name == 'nt':  # Windows
                            subprocess.run(f'rmdir /s /q "{temp_dir}"', shell=True, capture_output=True)
                        else:  # Linux/Mac
                            subprocess.run(f'rm -rf "{temp_dir}"', shell=True, capture_output=True)
                        
                        # Ждем немного и проверяем снова
                        await asyncio.sleep(2)
                        
                        if os.path.exists(temp_dir):
                            return False, f"❌ Не удалось удалить существующую директорию {temp_dir}"
                            
                except Exception as e:
                    logger.error(f"Ошибка удаления директории {temp_dir}: {e}")
                    return False, f"❌ Ошибка удаления временной директории: {str(e)}"
    
            # Ждем немного чтобы система освободила ресурсы
            await asyncio.sleep(1)
            
            # Проверяем что директории действительно нет
            if os.path.exists(temp_dir):
                return False, f"❌ Директория {temp_dir} все еще существует после удаления"
            
            # Клонируем репозиторий
            logger.info(f"Клонируем репозиторий {repo_url} ветка {branch}")
            process = await asyncio.create_subprocess_exec(
                "git", "clone", "-b", branch, "--depth", "1", repo_url, temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else stdout.decode()
                logger.error(f"Ошибка клонирования: {error_msg}")
                
                # Очищаем директорию в случае ошибки
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
                
                return False, f"❌ Ошибка клонирования: {error_msg}"
            
            # Создаем бэкап текущего кода
            backup_success, backup_msg = await self.create_backup()
            if not backup_success:
                # Очищаем временную директорию
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                return False, f"❌ Не удалось создать бэкап: {backup_msg}"
            
            # Копируем новые файлы (исключая конфигурационные)
            exclude_files = {'bot_database.db', 'self_learning_model.pth', 'config.json', 'training_datasets', 'backups', 'user_dialogues'}
            
            copied_files = []
            for item in os.listdir(temp_dir):
                if item in exclude_files or item.startswith('.'):
                    continue
                    
                src_path = os.path.join(temp_dir, item)
                dst_path = os.path.join('.', item)
                
                try:
                    if os.path.isdir(src_path):
                        if os.path.exists(dst_path):
                            logger.info(f"Удаляем существующую директорию {dst_path}")
                            shutil.rmtree(dst_path, ignore_errors=True)
                        logger.info(f"Копируем директорию {item}")
                        shutil.copytree(src_path, dst_path)
                    else:
                        logger.info(f"Копируем файл {item}")
                        shutil.copy2(src_path, dst_path)
                    
                    copied_files.append(item)
                    
                except Exception as e:
                    logger.error(f"Ошибка копирования {item}: {e}")
                    # Продолжаем копирование других файлов
            
            logger.info(f"Скопировано файлов: {len(copied_files)}")
            
            # Очищаем временную директорию
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Не удалось очистить временную директорию: {e}")
            
            if copied_files:
                return True, f"✅ Код успешно обновлен из GitHub. Скопировано файлов: {len(copied_files)}. Требуется перезагрузция."
            else:
                return False, "❌ Не удалось скопировать ни одного файла"
        
        except Exception as e:
            logger.error(f"Ошибка обновления кода: {e}")
            # Гарантированная очистка временной директории
            if os.path.exists("temp_update"):
                try:
                    shutil.rmtree("temp_update", ignore_errors=True)
                except:
                    pass
            return False, f"❌ Ошибка обновления: {str(e)}"


    async def show_system_status(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать статус системы"""
        status = await self.code_manager.get_system_status()
        
        if "error" in status:
            status_text = f"❌ Ошибка получения статуса: {status['error']}"
        else:
            system = status.get('system', {})
            bot = status.get('bot', {})
            
            status_text = (
                "📊 Статус системы\n\n"
                "🖥️ Система:\n"
                f"• Платформа: {system.get('platform', 'N/A')}\n"
                f"• Python: {system.get('python_version', 'N/A').split()[0]}\n"
                f"• Время работы: {system.get('bot_uptime', 'N/A')}\n"
                f"• CPU: {system.get('cpu_usage', 'N/A')}%\n"
                f"• Память: {system.get('memory_usage', 'N/A')}%\n"
                f"• Диск: {system.get('disk_usage', 'N/A')}%\n"
                f"• Процессы: {system.get('active_processes', 'N/A')}\n\n"
                "🤖 Бот:\n"
                f"• База данных: {bot.get('database_size', 'N/A')}\n"
                f"• Файлы логов: {bot.get('log_files_count', 'N/A')}\n"
                f"• Датасеты: {bot.get('training_datasets_count', 'N/A')}\n"
                f"• Модель ИИ: {bot.get('ai_model_status', 'N/A')}\n\n"
                f"🕐 {status.get('status', 'N/A')}\n"
                f"⏰ Обновлено: {status.get('timestamp', '')[:19]}"
            )
        
        await self.edit_message_with_cleanup(
            query, context,
            status_text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Обновить статус", callback_data="system_status")],
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )


    async def cancel_operation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Отмена любой операции"""
        context.user_data.clear()
        await self.send_message_with_cleanup(
            update, context, 
            "❌ Операция отменена",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
            ])
        )

    '''def setup_handlers(self):
        """Настройка обработчиков команд с новыми функциями"""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("menu", self.show_main_menu))
        self.application.add_handler(CommandHandler("admin", self.admin_panel))
        self.application.add_handler(CommandHandler("cancel", self.cancel_operation))
        self.application.add_handler(CommandHandler("ai", self.ai_chat))
        self.application.add_handler(CommandHandler("ai_stats", self.show_ai_stats))
        self.application.add_handler(CommandHandler("donate", self.show_donate))
        self.application.add_handler(CommandHandler("code_manager", self.code_manager_panel))
        self.application.add_handler(CommandHandler("force_learn", self.force_learning))
        
        # Обработчики сообщений для состояний
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message))
        self.application.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO | filters.VIDEO, self.handle_file_message))
        
        # Обработчики callback queries
        self.application.add_handler(CallbackQueryHandler(self.button_handler))

        self.application.add_handler(MessageHandler(
        filters.ALL, 
        self.fallback_handler
    ))'''

    async def show_helper(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать информацию о помощнике"""
        text = (
            f"{self.helper_text}\n\n"
            "Если вам нужна персональная помощь, свяжитесь с нашим помощником:"
        )
        
        keyboard = [
            [InlineKeyboardButton("💬 Написать помощнику", url=f"https://t.me/{self.helper_contact.replace('@', '')}")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]
        
        await self.edit_message_with_cleanup(
            query, context,
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def show_support(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать информацию о техподдержке и перенаправить в канал"""
        text = (
            "📞 Техническая поддержка\n\n"
            f"Если у вас возникли проблемы или вопросы, "
            f"обратитесь в нашу группу поддержки.\n\n"
            "Мы поможем вам решить любые проблемы!"
        )
        
        keyboard = [
            [InlineKeyboardButton("📞 Перейти в поддержку", url=f"https://t.me/{SUPPORT_GROUP_ID.replace('@', '')}")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]
        
        await self.edit_message_with_cleanup(
            query, context,
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def show_useful_info_safe(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Безопасная версия показа полезной информации"""
        try:
            # Проверяем доступность базы данных
            if not hasattr(self.db, 'get_all_useful_content'):
                await self.edit_message_with_cleanup(
                    query, context,
                    "ℹ️ Полезная информация\n\n"
                    "❌ Функция временно недоступна. База данных не инициализирована.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
                    ])
                )
                return
            
            # Пробуем получить данные
            useful_content = self.db.get_all_useful_content()
            
            if not useful_content:
                await self.edit_message_with_cleanup(
                    query, context,
                    "ℹ️ Полезная информация\n\n"
                    "Пока нет доступной полезной информации.\n\n"
                    "💡 Администратор может добавить информацию через админ-панель.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
                    ])
                )
                return
            
            # Создаем клавиатуру с контентом
            keyboard = []
            for content in useful_content:
                # Безопасно получаем название папки
                folder_name = content.get('folder_name', 'Общее')
                display_name = content['title'][:30] + "..." if len(content['title']) > 30 else content['title']
                
                keyboard.append([
                    InlineKeyboardButton(
                        f"📄 {display_name}", 
                        callback_data=f"download_useful_{content['id']}"
                    )
                ])
            
            keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
            
            await self.edit_message_with_cleanup(
                query, context,
                "ℹ️ Полезная информация\n\n"
                f"📁 Доступно материалов: {len(useful_content)}\n\n"
                "Выберите материал для скачивания:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Ошибка в show_useful_info_safe: {e}")
            await self.edit_message_with_cleanup(
                query, context,
                "❌ Произошла ошибка при загрузке полезной информации.\n\n"
                "Попробуйте позже или обратитесь к администратору.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
                ])
            )
            
    async def fallback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик для любых непонятных сообщений"""
        try:
            if update.message and update.message.text:
                text = update.message.text.strip()
                if text in ['/start', 'start', 'старт']:
                    await self.start(update, context)
                else:
                    await self.show_main_menu(update, context)
            else:
                # Обработка других типов сообщений
                await self.show_main_menu(update, context)
        except Exception as e:
            logger.error(f"Ошибка в fallback_handler: {e}")
            # Просто показываем меню в случае ошибки
            await self.show_main_menu(update, context)

    # =============================================================================
    # БЕЗОПАСНЫЕ МЕТОДЫ ДЛЯ УПРАВЛЕНИЯ ПОЛЕЗНОЙ ИНФОРМАЦИЕЙ
    # =============================================================================

    async def manage_useful_info_safe(self, query, context):
        """Безопасная версия manage_useful_info"""
        try:
            if hasattr(self.db, 'get_all_useful_content'):
                await self.manage_useful_info(query, context)
            else:
                await query.edit_message_text("❌ Функция управления полезной информацией временно недоступна")
        except Exception as e:
            logger.error(f"Ошибка в manage_useful_info_safe: {e}")
            await query.edit_message_text("❌ Ошибка при управлении полезной информацией")

    async def start_upload_useful_info_safe(self, query, context):
        """Безопасная версия start_upload_useful_info"""
        try:
            if hasattr(self.db, 'add_useful_content'):
                await self.start_upload_useful_info(query, context)
            else:
                await query.edit_message_text("❌ Функция загрузки полезной информации временно недоступна")
        except Exception as e:
            logger.error(f"Ошибка в start_upload_useful_info_safe: {e}")
            await query.edit_message_text("❌ Ошибка при загрузке полезной информации")

    async def show_useful_info_list_safe(self, query, context):
        """Безопасная версия show_useful_info_list"""
        try:
            if hasattr(self.db, 'get_all_useful_content'):
                await self.show_useful_info_list(query, context)
            else:
                await query.edit_message_text("❌ Функция просмотра полезной информации временно недоступна")
        except Exception as e:
            logger.error(f"Ошибка в show_useful_info_list_safe: {e}")
            await query.edit_message_text("❌ Ошибка при получении полезной информации")

    async def cleanup_previous_messages(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Удаление предыдущих сообщений бота"""
        try:
            if 'bot_message_ids' not in context.user_data:
                context.user_data['bot_message_ids'] = []
            
            bot_message_ids = context.user_data['bot_message_ids']
            
            while len(bot_message_ids) > 2:
                old_message_id = bot_message_ids.pop(0)
                try:
                    await context.bot.delete_message(
                        chat_id=update.effective_chat.id,
                        message_id=old_message_id
                    )
                except Exception as e:
                    logger.debug(f"Не удалось удалить сообщение {old_message_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Ошибка при очистке предыдущих сообщений: {e}")

    async def send_message_with_cleanup(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs):
        """Отправка сообщения с автоматической очисткой предыдущих"""
        await self.cleanup_previous_messages(update, context)
        
        if update.callback_query:
            message = await update.callback_query.message.reply_text(text, **kwargs)
        else:
            message = await update.message.reply_text(text, **kwargs)
        
        if 'bot_message_ids' not in context.user_data:
            context.user_data['bot_message_ids'] = []
        
        context.user_data['bot_message_ids'].append(message.message_id)
        
        return message

    async def edit_message_with_cleanup(self, query, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs):
        """Редактирование сообщения с автоматической очисткой предыдущих"""
        try:
            await query.edit_message_text(text, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка при редактировании сообщения: {e}")
            fake_update = Update(update_id=0, message=query.message)
            await self.send_message_with_cleanup(fake_update, context, "❌ Лекция не найдена в базе данных")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Пользователь"
        
        # ПОЛНАЯ ОЧИСТКА СОСТОЯНИЯ ПОЛЬЗОВАТЕЛЯ
        context.user_data.clear()
        
        # Добавляем пользователя в базу
        self.db.add_user(user_id, username)
        
        # Логируем запуск бота
        logger.info(f"Пользователь {user_id} ({username}) запустил бота")
        
        # Создаем клавиатуру главного меню
        keyboard = [
            [InlineKeyboardButton("🤖 ИИ-помощник", callback_data="ai_assistant")],
            [InlineKeyboardButton("📚 Предметы", callback_data="subjects")],
            [InlineKeyboardButton("📅 Расписание", callback_data="schedule")],
            [InlineKeyboardButton("ℹ️ Полезная информация", callback_data="useful_info")],
            [InlineKeyboardButton("👤 Ваш помощник", callback_data="helper")],
            [InlineKeyboardButton("📞 Техподдержка", callback_data="support")],
            [InlineKeyboardButton("💝 Пожертвования (в разработке)", callback_data="donate")],
        ]
        
        if user_id in ADMIN_IDS:
            keyboard.append([InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")])
            keyboard.append([InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Отправляем приветственное сообщение с главным меню
        welcome_text = (
            "👋 Добро пожаловать!\n\n"
            "У меня есть множество полезных функций!\n\n"
            "Выберите действие в меню ниже:"
        )
    
        await self.send_message_with_cleanup(update, context, welcome_text, reply_markup=reply_markup)

    async def show_main_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать главное меню"""
        keyboard = [
            [InlineKeyboardButton("🤖 ИИ-помощник", callback_data="ai_assistant")],
            [InlineKeyboardButton("📚 Предметы", callback_data="subjects")],
            [InlineKeyboardButton("📅 Расписание", callback_data="schedule")],
            [InlineKeyboardButton("ℹ️ Полезная информация", callback_data="useful_info")],
            [InlineKeyboardButton("👤 Ваш помощник", callback_data="helper")],
            [InlineKeyboardButton("📞 Техподдержка", callback_data="support")],
            [InlineKeyboardButton("💝 Пожертвования", callback_data="donate")],
        ]
        
        if update.effective_user.id in ADMIN_IDS:
            keyboard.append([InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")])
            keyboard.append([InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if hasattr(update, 'callback_query') and update.callback_query:
            await self.edit_message_with_cleanup(
                update.callback_query, context,
                "🏠 Главное меню\nПишите предложения по улучшению в Тех.поддержку\nВыберите действие:",
                reply_markup=reply_markup
            )
        else:
            await self.send_message_with_cleanup(
                update, context,
                "🏠 Главное меню\nПишите предложения по улучшению в Тех.поддержку\nВыберите действие:",
                reply_markup=reply_markup
            )

    async def show_donate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать информацию о пожертвованиях"""
        donate_text = (
            "💝 Поддержка проекта\n\n"
            "Эта функция находится в разработке.\n\n"
            "В будущем здесь можно будет поддержать развитие бота:\n"
            "• 💰 Пожертвования\n"
            "• 🚀 Приоритетная поддержка\n\n"
            "Следите за обновлениями!"
        )
        
        keyboard = [
            [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
        ]
        
        if update.callback_query:
            await self.edit_message_with_cleanup(
                update.callback_query, context,
                donate_text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await self.send_message_with_cleanup(
                update, context,
                donate_text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

    async def code_manager_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Панель управления кодом"""
        if update.effective_user.id not in ADMIN_IDS:
            await self.send_message_with_cleanup(update, context, "❌ У вас нет доступа к этой команде")
            return
        
        keyboard = [
            [InlineKeyboardButton("📊 Статус системы", callback_data="system_status")],
            [InlineKeyboardButton("🔄 Обновить код из GitHub", callback_data="update_code")],
            [InlineKeyboardButton("🧹 Очистить временные файлы", callback_data="cleanup_temp")],  # Новая кнопка
            [InlineKeyboardButton("🔄 Перезапустить бота", callback_data="restart_bot")],
            [InlineKeyboardButton("📁 Просмотр файлов", callback_data="view_files")],
            [InlineKeyboardButton("⚙️ Выполнить команду", callback_data="execute_command")],
            [InlineKeyboardButton("💾 Создать бэкап", callback_data="create_backup")],
            [InlineKeyboardButton("📋 Список бэкапов", callback_data="list_backups")],
            [InlineKeyboardButton("🤖 Принудительное обучение ИИ", callback_data="force_learning")],
            [InlineKeyboardButton("🔙 Админ-панель", callback_data="admin_panel")]
        ]
        
        if update.callback_query:
            await self.edit_message_with_cleanup(
                update.callback_query, context,
                "🔧 Управление кодом и системой\n\n"
                "Выберите действие:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await self.send_message_with_cleanup(
                update, context,
                "🔧 Управление кодом и системой\n\n"
                "Выберите действие:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

    async def force_learning(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Принудительное обучение ИИ из диалогов"""
        if update.effective_user.id not in ADMIN_IDS:
            await self.send_message_with_cleanup(update, context, "❌ У вас нет доступа к этой команде")
            return
        
        await self.send_message_with_cleanup(
            update, context,
            "🔄 Запуск принудительного обучения ИИ из диалогов пользователей...\n\n"
            "Это может занять несколько минут."
        )
        
        success, message = await self.ai_assistant.force_learning_from_dialogues()
        
        await self.send_message_with_cleanup(
            update, context,
            message,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )

    
    async def admin_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать админ-панель"""
        if update.effective_user.id not in ADMIN_IDS:
            await self.send_message_with_cleanup(update, context, "❌ У вас нет доступа к этой команде")
            return
        
        keyboard = [
            [InlineKeyboardButton("🔍 Диагностика обучения", callback_data="diagnose_training")],
            [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
            [InlineKeyboardButton("📤 Одиночная загрузка", callback_data="upload_file")],
            [InlineKeyboardButton("📚 Массовая загрузка", callback_data="mass_upload")],  # ← ЭТА КНОПКА
            [InlineKeyboardButton("🗑️ Удаление файлов", callback_data="delete_files")],   # ← И ЭТА
            [InlineKeyboardButton("➕ Добавить предмет", callback_data="add_subject")],
            [InlineKeyboardButton("👨‍🏫 Добавить преподавателя", callback_data="add_teacher")],
            [InlineKeyboardButton("📅 Управление расписанием", callback_data="manage_schedule")],
            [InlineKeyboardButton("📦 Управление полезной инфо", callback_data="manage_useful_info")],
            [InlineKeyboardButton("📋 Просмотр логов", callback_data="view_logs")],
            [InlineKeyboardButton("🤖 Статистика ИИ", callback_data="ai_stats")],
            [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
        ]
        
        if update.callback_query:
            await self.edit_message_with_cleanup(
                update.callback_query, context,
                "⚙️ Админ-панель\nВыберите действие:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await self.send_message_with_cleanup(
                update, context,
                "⚙️ Админ-панель\nВыберите действие:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )



    def setup_handlers(self):
        """Настройка обработчиков команд с новыми функциями"""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("menu", self.show_main_menu))
        self.application.add_handler(CommandHandler("admin", self.admin_panel))
        self.application.add_handler(CommandHandler("cancel", self.cancel_operation))
        self.application.add_handler(CommandHandler("ai", self.ai_chat))
        self.application.add_handler(CommandHandler("ai_stats", self.show_ai_stats))
        self.application.add_handler(CommandHandler("donate", self.show_donate))
        self.application.add_handler(CommandHandler("code_manager", self.code_manager_panel))
        self.application.add_handler(CommandHandler("force_learn", self.force_learning))
        
        # Обработчики сообщений для состояний
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message))
        self.application.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO | filters.VIDEO, self.handle_file_message))
        
        # Обработчики callback queries
        self.application.add_handler(CallbackQueryHandler(self.button_handler))

        self.application.add_handler(MessageHandler(
        filters.ALL, 
        self.fallback_handler
    ))
    
    # =============================================================================
    # ДОБАВЛЕННЫЕ CALLBACK ОБРАБОТЧИКИ
    # =============================================================================

    async def show_admin_panel_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для админ-панели"""
        await self.show_admin_panel(query, context)

    async def show_subjects_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для предметов"""
        await self.show_subjects(query, context)

    async def show_schedule_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для расписания"""
        await self.show_schedule(query, context)

    async def show_helper_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для помощника"""
        await self.show_helper(query, context)

    async def show_useful_info_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для полезной информации"""
        await self.show_useful_info_list_safe(query, context)  # Используем безопасную версию

    async def show_support_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для поддержки"""
        await self.show_support(query, context)

    async def back_to_menu_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для возврата в меню"""
        await self.show_main_menu(Update(update_id=0, callback_query=query), context)

    async def train_dataset_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для обучения на датасете"""
        await self.show_dataset_training(query, context)

    async def upload_dataset_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для загрузки датасета"""
        await self.start_upload_dataset(query, context)

    async def upload_github_dataset_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для загрузки датасета с GitHub"""
        await self.start_upload_github_dataset(query, context)

    async def manage_datasets_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для управления датасетами"""
        await self.show_manage_datasets(query, context)

    async def view_logs_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для просмотра логов"""
        await self.show_logs(query, context)

    async def manage_useful_info_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для управления полезной информацией"""
        await self.manage_useful_info_safe(query, context)  # Используем безопасную версию

    async def upload_useful_info_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для загрузки полезной информации"""
        await self.start_upload_useful_info_safe(query, context)  # Используем безопасную версию

    async def view_useful_info_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для просмотра полезной информации"""
        await self.show_useful_info_list_safe(query, context)  # Используем безопасную версию

    async def add_subject_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для добавления предмета"""
        await self.start_add_subject(query, context)

    async def add_teacher_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для добавления преподавателя"""
        await self.start_add_teacher(query, context)

    async def upload_file_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для одиночной загрузки"""
        await self.start_single_upload(query, context)

    async def manage_schedule_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для управления расписанием"""
        await self.manage_schedule(query, context)

    async def upload_schedule_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для загрузки расписания"""
        await self.start_upload_schedule(query, context)

    async def view_schedule_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для просмотра расписания"""
        await self.show_schedule_list(query, context)

    async def show_donate_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для кнопки пожертвований"""
        await self.show_donate(Update(update_id=0, callback_query=query), context)

    async def code_manager_panel_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Callback для панели управления кодом"""
        await self.code_manager_panel(Update(update_id=0, callback_query=query), context)

    async def ai_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /ai для прямого доступа к чату"""
        if 'ai_conversation' not in context.user_data:
            context.user_data['ai_conversation'] = []

        stats = self.ai_assistant.get_stats()

        welcome_text = (
            "🤖 ИИ-помощник (EnhancedSelfLearningAI)\n\n"
            "Привет! Задайте ваш вопрос, и я постараюсь помочь!\n\n"
            "💡 Используйте /cancel для выхода из чата"
        )

        keyboard = [
            [InlineKeyboardButton("🧹 Очистить историю", callback_data="ai_clear_history")],
            [InlineKeyboardButton("📊 Статистика ИИ", callback_data="ai_stats")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
        ]

        if update.effective_user.id in ADMIN_IDS:
            keyboard.insert(0, [InlineKeyboardButton("📚 Обучить на датасете", callback_data="train_dataset")])

        await self.send_message_with_cleanup(
            update, context,
            welcome_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

        context.user_data['state'] = 'ai_chat'

    async def show_ai_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать статистику ИИ"""
        stats = self.ai_assistant.get_stats()
        
        stats_text = (
            "📊 Статистика ИИ-помощника\n\n"
            f"🤖 Модель: {stats['current_model']}\n"
            f"👥 Пользователей: {stats['total_users']}\n"
            f"💬 Сообщений: {stats['total_messages']}\n"
            f"⚙️ Статус: {'✅ Активен' if stats['is_configured'] else '❌ Ошибка'}"
        )
        
        keyboard = [
            [InlineKeyboardButton("📚 Обучить на датасете", callback_data="train_dataset")],
            [InlineKeyboardButton("🤖 Чат с ИИ", callback_data="ai_assistant")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]
        
        await self.send_message_with_cleanup(
            update, context,
            stats_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cleanup_temp_files_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Очистка временных файлов через callback"""
        await self.edit_message_with_cleanup(
            query, context,
            "🧹 Очистка временных файлов...\n\n"
            "Пожалуйста, подождите."
        )
        
        success, message = await self.code_manager.cleanup_temp_files()
        
        await self.edit_message_with_cleanup(
            query, context,
            message,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
#============================================================
#===============Класс MassUploadHandler======================
#============================================================
    

   

# Определяем состояния ВНЕ класса
UPLOAD_FILES, SELECT_SUBJECT, SELECT_TYPE, CONFIRM_UPLOAD = range(4)

class MassUploadHandler:
    def __init__(self, database, file_manager):
        self.db = database
        self.file_manager = file_manager
        self.temp_uploads = {}
        
        self.SUPPORTED_EXTENSIONS = {
            'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt',
            'ppt', 'pptx', 'odp',
            'xls', 'xlsx', 'ods', 'csv',
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg',
            'zip', 'rar', '7z', 'tar', 'gz',
            'py', 'java', 'cpp', 'c', 'html', 'css', 'js', 'php', 'sql',
            'mp4', 'avi', 'mov', 'mkv', 'webm',
            'mp3', 'wav', 'ogg',
            'json', 'xml', 'yml', 'yaml'
        }

    def _get_file_extension(self, file_name: str) -> str:
        """Получает расширение файла в нижнем регистре без точки"""
        return Path(file_name).suffix.lower().lstrip('.')

    def _is_extension_supported(self, extension: str) -> bool:
        """Проверяет поддержку расширения файла"""
        return extension in self.SUPPORTED_EXTENSIONS

    async def start_mass_upload_simple(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Упрощенная массовая загрузка файлов"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        # Инициализация данных загрузки
        user_id = query.from_user.id
        self.temp_uploads[user_id] = {
            'files': [],
            'message_id': query.message.message_id
        }
        
        await query.edit_message_text(
            "📚 Массовая загрузка файлов\n\n"
            "Просто отправляйте файлы один за другим.\n"
            "Когда закончите - нажмите '✅ Завершить'\n\n"
            "❌ Для отмены используйте /cancel",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Завершить загрузку", callback_data="finish_mass_upload")],
                [InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")]
            ])
        )
        return UPLOAD_FILES

    async def handle_file_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка загружаемых файлов"""
        user_id = update.message.from_user.id
        
        if user_id not in self.temp_uploads:
            await update.message.reply_text("❌ Начните загрузку заново командой /mass_upload")
            return ConversationHandler.END

        upload_data = self.temp_uploads[user_id]
        
        # Определяем тип файла и получаем информацию
        file_info = None
        file_name = ""
        file_extension = ""
        
        if update.message.document:
            file_info = update.message.document
            file_name = file_info.file_name
            file_extension = self._get_file_extension(file_name)
        elif update.message.photo:
            file_info = update.message.photo[-1]
            file_name = f"photo_{file_info.file_id}.jpg"
            file_extension = "jpg"
        elif update.message.video:
            file_info = update.message.video
            file_name = getattr(file_info, 'file_name', f"video_{file_info.file_id}.mp4")
            file_extension = self._get_file_extension(file_name) or "mp4"
        elif update.message.audio:
            file_info = update.message.audio
            file_name = getattr(file_info, 'file_name', f"audio_{file_info.file_id}.mp3")
            file_extension = self._get_file_extension(file_name) or "mp3"
        else:
            await update.message.reply_text("❌ Этот тип файла не поддерживается")
            return UPLOAD_FILES

        # Проверяем расширение
        if not self._is_extension_supported(file_extension):
            await update.message.reply_text(
                f"❌ Файл '{file_name}' не поддерживается!\n"
                f"Расширение .{file_extension} не разрешено."
            )
            return UPLOAD_FILES

        # Добавляем файл в список
        upload_data['files'].append({
            'file_id': file_info.file_id,
            'file_name': file_name,
            'file_size': file_info.file_size,
            'file_extension': file_extension,
            'message_id': update.message.message_id
        })

        file_count = len(upload_data['files'])
        await update.message.reply_text(
            f"✅ Файл '{file_name}' добавлен!\n"
            f"📊 Всего файлов: {file_count}\n\n"
            f"Продолжайте отправлять файлы или нажмите '✅ Завершить'"
        )

        return UPLOAD_FILES

    async def finish_upload(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Завершение загрузки и выбор предмета"""
        user_id = query.from_user.id
        upload_data = self.temp_uploads.get(user_id)
        
        if not upload_data or not upload_data['files']:
            await query.answer("❌ Нет файлов для загрузки", show_alert=True)
            return UPLOAD_FILES

        # Получаем список предметов
        subjects = self.db.get_all_subjects()
        if not subjects:
            await query.answer("❌ Нет доступных предметов", show_alert=True)
            return UPLOAD_FILES

        # Создаем клавиатуру с предметами
        keyboard = []
        for subject in subjects:
            keyboard.append([InlineKeyboardButton(
                subject['name'], 
                callback_data=f"mass_subject_{subject['id']}"
            )])
        
        keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="cancel_mass_upload")])

        files_list = "\n".join([f"• {f['file_name']}" for f in upload_data['files'][:5]])
        if len(upload_data['files']) > 5:
            files_list += f"\n• ... и еще {len(upload_data['files']) - 5} файлов"

        await query.edit_message_text(
            f"📚 Массовая загрузка\n\n"
            f"📊 Файлов: {len(upload_data['files'])}\n"
            f"📄 Файлы:\n{files_list}\n\n"
            f"Выберите предмет для загрузки:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SELECT_SUBJECT

    async def select_subject(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Выбор предмета"""
        user_id = query.from_user.id
        subject_id = int(query.data.split('_')[-1])
        
        upload_data = self.temp_uploads.get(user_id)
        if not upload_data:
            await query.answer("❌ Ошибка загрузки", show_alert=True)
            return ConversationHandler.END

        upload_data['subject_id'] = subject_id
        subject_name = self.db.get_subject_name(subject_id)

        await query.edit_message_text(
            f"📚 Массовая загрузка\n\n"
            f"📝 Предмет: {subject_name}\n"
            f"📊 Файлов: {len(upload_data['files'])}\n\n"
            f"Выберите тип материалов:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📖 Лекции", callback_data="mass_type_lecture")],
                [InlineKeyboardButton("📝 Практические", callback_data="mass_type_practice")],
                [InlineKeyboardButton("📚 Доп. материалы", callback_data="mass_type_material")],
                [InlineKeyboardButton("🔙 Назад", callback_data="mass_back_subject")],
                [InlineKeyboardButton("❌ Отмена", callback_data="cancel_mass_upload")]
            ])
        )
        return SELECT_TYPE

    async def select_type(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Выбор типа материалов"""
        user_id = query.from_user.id
        file_type = query.data.split('_')[-1]
        
        upload_data = self.temp_uploads.get(user_id)
        if not upload_data:
            await query.answer("❌ Ошибка загрузки", show_alert=True)
            return ConversationHandler.END

        upload_data['file_type'] = file_type
        subject_name = self.db.get_subject_name(upload_data['subject_id'])
        type_names = {'lecture': 'Лекции', 'practice': 'Практические', 'material': 'Доп. материалы'}

        files_list = "\n".join([f"• {f['file_name']}" for f in upload_data['files'][:5]])
        if len(upload_data['files']) > 5:
            files_list += f"\n• ... и еще {len(upload_data['files']) - 5} файлов"

        await query.edit_message_text(
            f"📚 Массовая загрузка\n\n"
            f"📝 Предмет: {subject_name}\n"
            f"📁 Тип: {type_names[file_type]}\n"
            f"📊 Файлов: {len(upload_data['files'])}\n\n"
            f"📄 Файлы:\n{files_list}\n\n"
            f"Подтвердите загрузку:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Начать загрузку", callback_data="mass_upload_confirm")],
                [InlineKeyboardButton("🔙 Назад", callback_data="mass_back_type")],
                [InlineKeyboardButton("❌ Отмена", callback_data="cancel_mass_upload")]
            ])
        )
        return CONFIRM_UPLOAD

    async def confirm_upload(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Финальное подтверждение и загрузка файлов"""
        user_id = query.from_user.id
        upload_data = self.temp_uploads.get(user_id)
        
        if not upload_data:
            await query.answer("❌ Ошибка загрузки", show_alert=True)
            return ConversationHandler.END

        await query.edit_message_text(
            f"🔄 Начинаю загрузку...\n\n"
            f"📊 Файлов: {len(upload_data['files'])}\n"
            f"⏳ Это займет некоторое время...",
            reply_markup=None
        )

        success_count = 0
        error_count = 0
        errors = []

        # Загружаем файлы по одному
        for i, file_data in enumerate(upload_data['files'], 1):
            try:
                # Обновляем прогресс каждые 5 файлов
                if i % 5 == 0 or i == len(upload_data['files']):
                    progress = int((i / len(upload_data['files'])) * 20)
                    progress_bar = "[" + "█" * progress + "▒" * (20 - progress) + "]"
                    
                    await query.edit_message_text(
                        f"🔄 Загружаю файлы...\n\n"
                        f"📊 Прогресс: {i}/{len(upload_data['files'])}\n"
                        f"{progress_bar} {int((i/len(upload_data['files']))*100)}%\n\n"
                        f"✅ Успешно: {success_count}\n"
                        f"❌ Ошибок: {error_count}",
                        reply_markup=None
                    )

                # Загружаем файл
                result = await self.file_manager.upload_file(
                    file_data['file_id'],
                    upload_data['subject_id'],
                    upload_data['file_type'],
                    file_data['file_name'],
                    user_id
                )

                if result['success']:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"{file_data['file_name']}: {result.get('error', 'Неизвестная ошибка')}")

                # Небольшая задержка чтобы не перегружать сервер
                await asyncio.sleep(0.3)

            except Exception as e:
                error_count += 1
                errors.append(f"{file_data['file_name']}: {str(e)}")

        # Формируем итоговое сообщение
        message = f"✅ Загрузка завершена!\n\n"
        message += f"📊 Результаты:\n"
        message += f"✅ Успешно: {success_count}\n"
        message += f"❌ Ошибок: {error_count}\n"

        if errors and error_count > 0:
            error_list = "\n".join(errors[:3])
            if error_count > 3:
                error_list += f"\n... и еще {error_count - 3} ошибок"
            message += f"\n❌ Ошибки:\n{error_list}"

        # Очищаем временные данные
        if user_id in self.temp_uploads:
            del self.temp_uploads[user_id]

        await query.edit_message_text(
            message,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📤 Новая загрузка", callback_data="mass_upload_start")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )

        return ConversationHandler.END

    async def navigate_back(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Навигация назад"""
        user_id = query.from_user.id
        back_to = query.data.split('_')[-1]
        
        if back_to == 'subject':
            return await self.finish_upload(query, context)
        elif back_to == 'type':
            upload_data = self.temp_uploads.get(user_id)
            if upload_data and upload_data['subject_id']:
                # Возвращаемся к выбору предмета
                temp_query = type('obj', (object,), {
                    'from_user': query.from_user,
                    'message': query.message,
                    'data': f"mass_subject_{upload_data['subject_id']}"
                })()
                return await self.select_subject(temp_query, context)
        
        return ConversationHandler.END

    async def cancel_upload(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Отмена загрузки"""
        user_id = query.from_user.id
        
        if user_id in self.temp_uploads:
            del self.temp_uploads[user_id]
        
        await query.edit_message_text(
            "❌ Массовая загрузка отменена.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
        return ConversationHandler.END


# Функция setup должна быть ВНЕ класса
def setup_mass_upload_handlers(application, mass_upload_handler):
    """Настройка обработчиков массовой загрузки"""
    mass_upload_conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(mass_upload_handler.start_mass_upload_simple, pattern="^mass_upload$"),
            CallbackQueryHandler(mass_upload_handler.start_mass_upload_simple, pattern="^mass_upload_start$"),
            CallbackQueryHandler(mass_upload_handler.start_mass_upload_simple, pattern="^mass_upload_simple$")
        ],
        states={
            UPLOAD_FILES: [
                MessageHandler(
                    filters.DOCUMENT | filters.PHOTO | filters.VIDEO | filters.AUDIO,
                    mass_upload_handler.handle_file_upload
                ),
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    mass_upload_handler.handle_text_message  # ← ДОБАВЛЕНО
                ),
                CallbackQueryHandler(mass_upload_handler.finish_upload, pattern="^finish_mass_upload$")
            ],
            SELECT_SUBJECT: [
                CallbackQueryHandler(mass_upload_handler.select_subject, pattern="^mass_subject_"),
                CallbackQueryHandler(mass_upload_handler.navigate_back, pattern="^mass_back_subject$")
            ],
            SELECT_TYPE: [
                CallbackQueryHandler(mass_upload_handler.select_type, pattern="^mass_type_"),
                CallbackQueryHandler(mass_upload_handler.navigate_back, pattern="^mass_back_type$")
            ],
            CONFIRM_UPLOAD: [
                CallbackQueryHandler(mass_upload_handler.confirm_upload, pattern="^mass_upload_confirm$"),
                CallbackQueryHandler(mass_upload_handler.navigate_back, pattern="^mass_back_type$")
            ]
        },
        fallbacks=[
            CallbackQueryHandler(mass_upload_handler.cancel_upload, pattern="^cancel_mass_upload$"),
            CallbackQueryHandler(mass_upload_handler.cancel_upload, pattern="^admin_panel$"),
            CommandHandler("cancel", mass_upload_handler.cancel_upload)
        ],
        name="mass_upload",
        persistent=False
    )
    
    application.add_handler(mass_upload_conv)



    async def show_delete_files_menu(self, query, context: ContextTypes.DEFAULT_TYPE):
    
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        await self.edit_message_with_cleanup(
            query, context,
            "🗑️ Удаление файлов\n\n"
            "Выберите тип файлов для удаления:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📓 Лекции", callback_data="delete_lectures_menu")],
                [InlineKeyboardButton("📝 Практические", callback_data="delete_practices_menu")],
                [InlineKeyboardButton("📅 Расписания", callback_data="delete_schedules_menu")],
                [InlineKeyboardButton("📦 Полезная информация", callback_data="delete_useful_menu")],
                [InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")]
            ])
        )

    async def delete_lectures_menu(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Меню удаления лекций"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        subjects = self.db.get_all_subjects()
        
        if not subjects:
            await self.edit_message_with_cleanup(
                query, context,
                "❌ Нет предметов для удаления лекций",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="delete_files")]
                ])
            )
            return
        
        keyboard = []
        for subject in subjects:
            lectures_count = self.db.get_subject_lectures_count(subject['id'])
            keyboard.append([
                InlineKeyboardButton(
                    f"📓 {subject['name']} ({lectures_count} лекций)", 
                    callback_data=f"delete_lectures_subject_{subject['id']}"
                )
            ])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="delete_files")])
        
        await self.edit_message_with_cleanup(
            query, context,
            "🗑️ Удаление лекций\n\nВыберите предмет:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


    
    async def handle_ai_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений в AI-чате"""
        user_message = update.message.text.strip()
        
        if not user_message:
            await self.send_message_with_cleanup(update, context, "Пожалуйста, введите ваш вопрос:")
            return

        await update.message.chat.send_action(action="typing")

        try:
            user_id = update.effective_user.id
            ai_response, success, model_used = await self.ai_assistant.get_ai_response(user_id, user_message)

            if 'ai_conversation' not in context.user_data:
                context.user_data['ai_conversation'] = []

            context.user_data['ai_conversation'].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ai_response}
            ])

            if len(context.user_data['ai_conversation']) > 10:
                context.user_data['ai_conversation'] = context.user_data['ai_conversation'][-10:]

            keyboard = [
                [InlineKeyboardButton("🧹 Очистить историю", callback_data="ai_clear_history")],
                [InlineKeyboardButton("📊 Статистика ИИ", callback_data="ai_stats")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
            ]

            if update.effective_user.id in ADMIN_IDS:
                keyboard.insert(0, [InlineKeyboardButton("📚 Обучить на датасете", callback_data="train_dataset")])

            await self.send_message_with_cleanup(
                update, context,
                f"🤖 {ai_response}",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )

        except Exception as e:
            logger.error(f"Ошибка в AI чате: {e}")
            await self.send_message_with_cleanup(
                update, context,
                "❌ Произошла ошибка при обработке вашего запроса. Попробуйте позже.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
                ])
            )

    



    async def handle_file_viewing(self, update: Update, context: ContextTypes.DEFAULT_TYPE, file_path: str):
        """Обработка просмотра файлов"""
        success, content, message = self.code_manager.view_file(file_path)
        
        if success:
            # Обрезаем длинный контент для Telegram
            if len(content) > 4000:
                content = content[:4000] + "\n\n... (файл обрезан, слишком длинный)"
            
            response_text = f"📁 {message}\n\n```\n{content}\n```"
        else:
            response_text = message
        
        await self.send_message_with_cleanup(
            update, context,
            response_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📁 Просмотреть другой файл", callback_data="view_files")],
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")]
            ])
        )
        
        context.user_data.clear()

    async def handle_command_execution(self, update: Update, context: ContextTypes.DEFAULT_TYPE, command: str):
        """Обработка выполнения команд"""
        # Исправленный вызов
        success, output, message = await self.code_manager.execute_command(command)
            
        response_text = f"⚙️ {message}\n\n"
        
        if success:
            response_text += "✅ Команда выполнена успешно\n\n"
        else:
            response_text += "❌ Ошибка выполнения команды\n\n"
        
        if output:
            # Обрезаем длинный вывод
            if len(output) > 3500:
                output = output[:3500] + "\n\n... (вывод обрезан, слишком длинный)"
            response_text += f"```\n{output}\n```"
        
        await self.send_message_with_cleanup(
            update, context,
            response_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Выполнить другую команду", callback_data="execute_command")],
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")]
            ])
        )
        
        context.user_data.clear()

    async def handle_restart_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, confirmation: str):
        """Обработка подтверждения перезапуска"""
        if confirmation.lower() in ['да', 'yes', 'y', 'confirm']:
            success, message = self.code_manager.restart_bot()
            await self.send_message_with_cleanup(update, context, message)
        else:
            await self.send_message_with_cleanup(
                update, context,
                "❌ Перезапуск отменен.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")]
                ])
            )
        
        context.user_data.clear()

    async def show_logs(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать логи системы с кликабельными кнопками"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        logs = self.db.get_logs(limit=50)
        
        if not logs:
            await self.edit_message_with_cleanup(
                query, context,
                "📋 Логи системы\n\nПока нет записей в логах.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📋 Все логи", callback_data="view_logs_all")],
                    [InlineKeyboardButton("❌ Ошибки", callback_data="view_logs_error")],
                    [InlineKeyboardButton("⚠️ Предупреждения", callback_data="view_logs_warning")],
                    [InlineKeyboardButton("ℹ️ Инфо", callback_data="view_logs_info")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")]
                ])
            )
            return
        
        # Группируем логи по уровням
        error_logs = [log for log in logs if log['level'] == 'ERROR']
        warning_logs = [log for log in logs if log['level'] == 'WARNING']
        info_logs = [log for log in logs if log['level'] == 'INFO']
        
        logs_text = (
            "📋 Логи системы\n\n"
            f"❌ Ошибки: {len(error_logs)}\n"
            f"⚠️ Предупреждения: {len(warning_logs)}\n"
            f"ℹ️ Инфо: {len(info_logs)}\n\n"
            "Нажмите на кнопку ниже для просмотра конкретного типа логов:"
        )
        
        await self.edit_message_with_cleanup(
            query, context,
            logs_text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📋 Все логи", callback_data="view_logs_all")],
                [InlineKeyboardButton("❌ Ошибки", callback_data="view_logs_error")],
                [InlineKeyboardButton("⚠️ Предупреждения", callback_data="view_logs_warning")],
                [InlineKeyboardButton("ℹ️ Инфо", callback_data="view_logs_info")],
                [InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")]
            ])
        )

    # =============================================================================
    # НОВЫЕ ФУНКЦИИ ДЛЯ УПРАВЛЕНИЯ ДАТАСЕТАМИ
    # =============================================================================

    async def show_manage_datasets(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать интерфейс управления датасетами"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        datasets = self.ai_assistant.get_datasets_info()
        
        if not datasets:
            await self.edit_message_with_cleanup(
                query, context,
                "📚 Управление датасетами\n\n"
                "Пока нет загруженных датасетов.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📤 Загрузить датасет", callback_data="upload_dataset")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")]
                ])
            )
            return
        
        # Сортируем датасеты по размеру (от большего к меньшему)
        datasets.sort(key=lambda x: x['size_bytes'], reverse=True)
        
        total_size = sum(d['size_bytes'] for d in datasets)
        total_files = len(datasets)
        
        message_text = (
            "📚 Управление датасетами\n\n"
            f"📊 Всего датасетов: {total_files}\n"
            f"💾 Общий размер: {total_size / (1024*1024):.1f} MB\n\n"
            "Доступные датасеты:\n"
        )
        
        # Добавляем информацию о каждом датасете
        for i, dataset in enumerate(datasets[:10]):  # Показываем первые 10
            message_text += f"\n{i+1}. {dataset['filename']} ({dataset['size_mb']} MB)"
        
        if total_files > 10:
            message_text += f"\n\n... и еще {total_files - 10} датасетов"
        
        keyboard = []
        
        # Добавляем кнопки для каждого датасета (обучение и удаление)
        for dataset in datasets[:5]:  # Показываем кнопки для первых 5 датасетов
            filename = dataset['filename']
            display_name = filename[:20] + "..." if len(filename) > 20 else filename
            
            keyboard.append([
                InlineKeyboardButton(f"🎯 Обучить: {display_name}", callback_data=f"train_on_dataset_{filename}"),
                InlineKeyboardButton(f"🗑️ Удалить", callback_data=f"delete_dataset_{filename}")
            ])
        
        # Если датасетов больше 5, добавляем кнопку "Показать все"
        if total_files > 5:
            keyboard.append([InlineKeyboardButton("📋 Показать все датасеты", callback_data="view_all_datasets")])
        
        keyboard.extend([
            [InlineKeyboardButton("📤 Загрузить новый датасет", callback_data="upload_dataset")],
            [InlineKeyboardButton("🐙 Загрузить с GitHub", callback_data="upload_github_dataset")],
            [InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")]
        ])
        
        await self.edit_message_with_cleanup(
            query, context,
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def delete_dataset(self, query, context: ContextTypes.DEFAULT_TYPE, dataset_name: str):
        """Удалить датасет"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        # Подтверждение удаления
        if not context.user_data.get(f'confirm_delete_{dataset_name}'):
            context.user_data[f'confirm_delete_{dataset_name}'] = True
            
            dataset_info = self.ai_assistant.dataset_manager.get_dataset_info(dataset_name)
            if dataset_info:
                size_info = f" ({dataset_info['size_mb']} MB, создан {dataset_info['days_old']} дней назад)"
            else:
                size_info = ""
            
            await self.edit_message_with_cleanup(
                query, context,
                f"⚠️ Подтверждение удаления\n\n"
                f"Вы уверены, что хотите удалить датасет:\n"
                f"`{dataset_name}`{size_info}\n\n"
                f"❌ Это действие нельзя отменить!",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Да, удалить", callback_data=f"delete_dataset_{dataset_name}")],
                    [InlineKeyboardButton("❌ Отмена", callback_data="manage_datasets")]
                ])
            )
            return
        
        # Выполнение удаления после подтверждения
        try:
            success, message = self.ai_assistant.delete_dataset(dataset_name)
            
            # Очищаем флаг подтверждения
            if f'confirm_delete_{dataset_name}' in context.user_data:
                del context.user_data[f'confirm_delete_{dataset_name}']
            
            if success:
                # Логируем удаление
                self.db.add_log("INFO", f"Датасет удален: {dataset_name}", query.from_user.id)
                
                await self.edit_message_with_cleanup(
                    query, context,
                    message,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
                        [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                    ])
                )
            else:
                await self.edit_message_with_cleanup(
                    query, context,
                    message,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔄 Попробовать снова", callback_data=f"delete_dataset_{dataset_name}")],
                        [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
                        [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                    ])
                )
                
        except Exception as e:
            logger.error(f"Ошибка при удалении датасета: {e}")
            await self.edit_message_with_cleanup(
                query, context,
                f"❌ Произошла ошибка при удалении датасета: {str(e)}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )

    async def start_dataset_training(self, query, context: ContextTypes.DEFAULT_TYPE, dataset_name: str):
        """Запуск обучения на датасете с улучшенным интерфейсом"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        # Получаем информацию о датасете
        dataset_info = self.ai_assistant.dataset_manager.get_dataset_info(dataset_name)
        size_info = f" ({dataset_info['size_mb']} MB)" if dataset_info else ""
        
        await self.edit_message_with_cleanup(
            query, context,
            f"🔄 Запуск обучения на датасете: {dataset_name}{size_info}\n\n"
            "📊 Процесс обучения:\n"
            "• Загрузка и проверка данных\n"
            "• Векторизация текстов\n" 
            "• Обучение нейросети\n"
            "• Сохранение модели\n\n"
            "⏳ Это может занять несколько минут...\n"
            "Бот продолжит работать во время обучения.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отмена", callback_data="manage_datasets")]
            ])
        )
        
        # Используем asyncio для асинхронного выполнения
        try:
            success, message = await self.ai_assistant.train_on_dataset(dataset_name)
            
            # Логируем результат обучения
            log_level = "INFO" if success else "ERROR"
            self.db.add_log(log_level, f"Обучение на датасете {dataset_name}: {'успех' if success else 'ошибка'}", query.from_user.id)
            
            # Создаем клавиатуру в зависимости от результата
            if success:
                keyboard = [
                    [InlineKeyboardButton("🤖 Чат с ИИ", callback_data="ai_assistant")],
                    [InlineKeyboardButton("📊 Статистика", callback_data="ai_stats")],
                    [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ]
            else:
                keyboard = [
                    [InlineKeyboardButton("🔄 Попробовать снова", callback_data=f"train_on_dataset_{dataset_name}")],
                    [InlineKeyboardButton("📚 Выбрать другой датасет", callback_data="manage_datasets")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ]
            
            await self.edit_message_with_cleanup(
                query, context,
                message,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обучении на датасете: {e}")
            await self.edit_message_with_cleanup(
                query, context,
                f"❌ Неожиданная ошибка при обучении: {str(e)}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Попробовать снова", callback_data=f"train_on_dataset_{dataset_name}")],
                    [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )

    # =============================================================================
    # ОБНОВЛЕННЫЙ ИНТЕРФЕЙС ОБУЧЕНИЯ НА ДАТАСЕТАХ
    # =============================================================================

    async def show_dataset_training(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать интерфейс обучения на датасетах с улучшенной навигацией"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        available_datasets = self.ai_assistant.get_datasets_info()
        supported_formats = self.ai_assistant.self_learning_ai.dataset_trainer.get_supported_formats()
        
        # Форматируем информацию о поддерживаемых форматах
        formats_text = "\n".join([f"• {fmt}" for fmt in supported_formats.values()])
        
        total_size = sum(d['size_bytes'] for d in available_datasets)
        
        keyboard = [
            [InlineKeyboardButton("📤 Загрузить датасет", callback_data="upload_dataset")],
            [InlineKeyboardButton("🐙 Загрузить с GitHub", callback_data="upload_github_dataset")],
            [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
            [InlineKeyboardButton("🔙 Назад", callback_data="ai_stats")]
        ]
        
        if available_datasets:
            # Показываем кнопки для обучения на доступных датасетах
            for dataset_info in available_datasets[:3]:  # Показываем первые 3
                btn_text = f"🎯 Обучить: {dataset_info['filename']} ({dataset_info['size_mb']} MB)"
                keyboard.insert(0, [
                    InlineKeyboardButton(
                        btn_text, 
                        callback_data=f"train_on_dataset_{dataset_info['filename']}"
                    )
                ])
        
        message_text = (
            "📚 Обучение на датасетах\n\n"
            f"📂 Доступно датасетов: {len(available_datasets)}\n"
            f"💾 Общий размер: {total_size / (1024*1024):.1f} MB\n\n"
            "📋 Поддерживаемые форматы:\n"
            f"{formats_text}\n\n"
            "🐙 Поддерживаются GitHub репозитории"
        )
        
        await self.edit_message_with_cleanup(
            query, context,
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    

    async def clear_ai_history(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Очистка истории AI-чата"""
        if 'ai_conversation' in context.user_data:
            context.user_data['ai_conversation'] = []
        
        self.ai_assistant.clear_conversation_history(query.from_user.id)
        
        await self.edit_message_with_cleanup(
            query, context,
            "🧹 История разговора очищена!\n\n"
            "Теперь вы можете начать новый диалог.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🤖 Задать вопрос", callback_data="ai_assistant")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
            ])
        )

    async def start_upload_dataset(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать загрузку датасета с информацией о ограничениях"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        context.user_data.clear()
        context.user_data['state'] = 'uploading_dataset'
        
        await self.edit_message_with_cleanup(
            query, context,
            "📤 Загрузка датасета\n\n"
            "📁 Поддерживаемые форматы:\n"
            "• JSON, JSONL (.json, .jsonl)\n"
            "• CSV, TSV (.csv, .tsv, .txt)\n"
            "• Excel (.xlsx, .xls)\n"
            "• Текстовые файлы (.txt, .md)\n"
            "• YAML (.yaml, .yml)\n"
            "• XML (.xml)\n"
            "• Parquet (.parquet)\n"
            "• Feather (.feather)\n"
            "• Pickle (.pkl, .pickle)\n\n"
            "⚠️ Ограничения Telegram:\n"
            "• Максимальный размер файла: 50MB\n"
            "• Для больших файлов используйте GitHub\n\n"
            "💡 Для файлов больше 50MB:\n"
            "• Используйте опцию 'Загрузить с GitHub'\n"
            "• Разделите файл на части\n"
            "• Используйте сжатие (.zip)\n\n"
            "Отправьте файл датасета:\n\n"
            "❌ Для отмены используйте /cancel",
            parse_mode='HTML'
        )

    async def start_upload_github_dataset(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать загрузку датасета с GitHub с информацией о новых форматах"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        context.user_data.clear()
        context.user_data['state'] = 'uploading_github_dataset'
        
        supported_formats = self.ai_assistant.self_learning_ai.dataset_trainer.get_supported_formats()
        
        formats_text = "📋 Поддерживаемые форматы:\n"
        for format_type, extensions in supported_formats.items():
            formats_text += f"• {format_type.upper()}: {', '.join(extensions)}\n"
        
        await self.edit_message_with_cleanup(
            query, context,
            "🐙 Загрузка датасета с GitHub\n\n"
            f"{formats_text}\n\n"
            "🔗 Примеры работающих ссылок:\n"
            "• <code>https://github.com/huggingface/datasets</code>\n"
            "• <code>https://github.com/username/repo/blob/main/data.json</code>\n"
            "• <code>https://raw.githubusercontent.com/username/repo/main/data.csv</code>\n"
            "• <code>https://github.com/username/repo/blob/main/dataset.xlsx</code>\n\n"
            "💡 Советы:\n"
            "• Убедитесь, что репозиторий публичный\n"
            "• Поддерживаются JSON, CSV, Excel, текстовые файлы и многие другие\n"
            "• Для больших репозиторий укажите прямой путь к файлу\n\n"
            "Отправьте ссылку на GitHub:",
            parse_mode='HTML'
        )

    

    async def save_dataset_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сохранение загруженного датасета с проверкой размера"""
        if not update.message.document:
            await self.send_message_with_cleanup(update, context, "❌ Пожалуйста, отправьте файл датасета.")
            return
        
        file = await update.message.document.get_file()
        filename = update.message.document.file_name
        file_size = update.message.document.file_size
        
        # Максимальный размер для Telegram - 50MB
        MAX_TELEGRAM_SIZE = 50 * 1024 * 1024  # 50MB в байтах
        
        if file_size and file_size > MAX_TELEGRAM_SIZE:
            await self.send_message_with_cleanup(
                update, context,
                f"❌ Файл слишком большой для загрузки через Telegram.\n\n"
                f"📊 Размер вашего файла: {file_size / (1024*1024):.1f}MB\n"
                f"📏 Максимальный размер: 50MB\n\n"
                "💡 Рекомендации:\n"
                "• Разделите большой датасет на несколько файлов\n"
                "• Используйте сжатые форматы (.zip, .gz)\n"
                "• Загрузите файл через GitHub и используйте ссылку"
            )
            return
        
        # Проверяем расширение файла
        supported_extensions = ['.json', '.csv', '.jsonl', '.xlsx', '.xls', '.txt', '.yaml', '.yml', '.xml', '.parquet', '.feather', '.pkl', '.pickle']
        if not any(filename.endswith(ext) for ext in supported_extensions):
            await self.send_message_with_cleanup(
                update, context,
                f"❌ Неподдерживаемый формат файла. Используйте: {', '.join(supported_extensions)}"
            )
            return
        
        try:
            # Создаем директорию для датасетов если не существует
            datasets_dir = "training_datasets"
            os.makedirs(datasets_dir, exist_ok=True)
            
            file_path = os.path.join(datasets_dir, filename)
            
            # Показываем прогресс загрузки для файлов больше 10MB
            if file_size and file_size > 10 * 1024 * 1024:
                progress_msg = await update.message.reply_text(
                    f"📥 Загрузка файла {filename}...\n"
                    f"Размер: {file_size / (1024*1024):.1f}MB\n"
                    "⏳ Пожалуйста, подождите..."
                )
            
            # Загружаем файл
            await file.download_to_drive(file_path)
            
            # Удаляем сообщение о прогрессе если было создано
            if file_size and file_size > 10 * 1024 * 1024:
                try:
                    await progress_msg.delete()
                except:
                    pass
            
            # Проверяем что файл успешно загружен
            if not os.path.exists(file_path):
                await self.send_message_with_cleanup(
                    update, context,
                    "❌ Ошибка при загрузке файла. Попробуйте еще раз."
                )
                return
            
            actual_size = os.path.getsize(file_path)
            logger.info(f"Файл {filename} успешно загружен, размер: {actual_size} байт")
            
            context.user_data.clear()
            
            await self.send_message_with_cleanup(
                update, context,
                f"✅ Датасет '{filename}' успешно загружен!\n\n"
                f"📊 Размер файла: {actual_size / (1024*1024):.1f}MB\n\n"
                "Теперь вы можете обучить ИИ на этом датасете.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📚 Обучить на этом датасете", callback_data=f"train_on_dataset_{filename}")],
                    [InlineKeyboardButton("📚 Выбрать другой датасет", callback_data="train_dataset")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении датасета: {e}")
            await self.send_message_with_cleanup(
                update, context,
                f"❌ Ошибка при сохранении датасета: {str(e)}"
            )
                
            context.user_data.clear()
                
    async def handle_github_url(self, update: Update, context: ContextTypes.DEFAULT_TYPE, github_url: str):
        """Обработка GitHub URL"""
        try:
            if not github_url.startswith('https://github.com') and not github_url.startswith('https://raw.githubusercontent.com'):
                await self.send_message_with_cleanup(
                    update, context,
                    "❌ Неверный GitHub URL. Используйте ссылки вида:\n"
                    "• https://github.com/user/repo\n"
                    "• https://github.com/user/repo/blob/main/file.json\n"
                    "• https://raw.githubusercontent.com/user/repo/main/file.csv"
                )
                return
            
            await update.message.chat.send_action(action="typing")
            
            # Скачиваем и обучаем
            success, message = await self.ai_assistant.train_from_github(github_url)
            
            await self.send_message_with_cleanup(
                update, context,
                message,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🤖 Чат с ИИ", callback_data="ai_assistant")],
                    [InlineKeyboardButton("📊 Статистика", callback_data="ai_stats")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
            
            context.user_data.clear()
            
        except Exception as e:
            logger.error(f"Ошибка при обработке GitHub URL: {e}")
            await self.send_message_with_cleanup(
                update, context,
                f"❌ Ошибка при загрузке с GitHub: {str(e)}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Попробовать снова", callback_data="upload_github_dataset")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )

    async def show_ai_stats_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать статистику ИИ через callback"""
        stats = self.ai_assistant.get_stats()
        
        stats_text = (
            "📊 Статистика ИИ-помощника\n\n"
            f"🤖 Модель: {stats['current_model']}\n"
            f"👥 Пользователей: {stats['total_users']}\n"
            f"💬 Сообщений: {stats['total_messages']}\n"
            f"⚙️ Статус: {'✅ Активен' if stats['is_configured'] else '❌ Ошибка'}"
        )
        
        keyboard = [
            [InlineKeyboardButton("📚 Обучить на датасете", callback_data="train_dataset")],
            [InlineKeyboardButton("🤖 Чат с ИИ", callback_data="ai_assistant")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]
        
        await self.edit_message_with_cleanup(
            query, context,
            stats_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    

    async def show_subject_content(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Показать контент предмета (лекции и практические)"""
        logger.info(f"Показать контент предмета: {subject_id}")
        
        subject = self.db.get_subject(subject_id)
        if not subject:
            await self.edit_message_with_cleanup(query, context, "❌ Предмет не найден")
            return
            
        lectures = self.db.get_lectures(subject_id)
        practices = self.db.get_practices(subject_id)
        
        logger.info(f"Лекций: {len(lectures)}, Практических: {len(practices)}")
        
        keyboard = []
        
        if lectures:
            keyboard.append([InlineKeyboardButton("📓 Лекции", callback_data=f"show_lectures_{subject_id}")])
        
        if practices:
            keyboard.append([InlineKeyboardButton("📝 Практические работы", callback_data=f"show_practices_{subject_id}")])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад к предметам", callback_data="subjects")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")])
        
        text = f"📖 {subject['name']}\n\n"
        text += f"📓 Лекций: {len(lectures)}\n"
        text += f"📝 Практических работ: {len(practices)}"
        
        await self.edit_message_with_cleanup(
            query, context,
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def manage_schedule(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Управление расписанием (админ)"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        schedules = self.db.get_all_schedule()
        
        keyboard = [
            [InlineKeyboardButton("📤 Загрузить расписание", callback_data="upload_schedule")],
            [InlineKeyboardButton("📋 Просмотреть расписание", callback_data="view_schedule")]
        ]
        
        if schedules:
            for schedule in schedules:
                keyboard.append([
                    InlineKeyboardButton(f"🗑️ Удалить: {schedule['title']}", callback_data=f"delete_schedule_{schedule['id']}")
                ])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")])
        
        await self.edit_message_with_cleanup(
            query, context,
            "📅 Управление расписанием\n\n"
            f"📊 Загружено файлов: {len(schedules)}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def start_single_upload(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать одиночную загрузку файла"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        subjects = self.db.get_all_subjects()
        
        if not subjects:
            await self.edit_message_with_cleanup(
                query, context,
                "📤 Одиночная загрузка файла\n\n"
                "❌ Сначала добавьте предметы.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Добавить предмет", callback_data="add_subject")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
            return
        
        context.user_data.clear()
        context.user_data['state'] = 'single_upload_subject'
        
        # Создаем клавиатуру с предметами
        keyboard = []
        for subject in subjects:
            keyboard.append([InlineKeyboardButton(
                f"📖 {subject['name']}", 
                callback_data=f"upload_subject_{subject['id']}"
            )])
        
        keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")])
        
        await self.edit_message_with_cleanup(
            query, context,
            "📤 Одиночная загрузка файла\n\n"
            "Выберите предмет для загрузки:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_select_upload_subject(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Обработка выбора предмета для загрузки"""
        context.user_data['upload_subject_id'] = subject_id
        context.user_data['state'] = 'single_upload_type'
        
        subject = self.db.get_subject(subject_id)
        
        keyboard = [
            [InlineKeyboardButton("📓 Лекция", callback_data="upload_type_lecture")],
            [InlineKeyboardButton("📝 Практическая работа", callback_data="upload_type_practice")],
            [InlineKeyboardButton("🔙 Назад к выбору предмета", callback_data="upload_file")],
            [InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")]
        ]
        
        await self.edit_message_with_cleanup(
            query, context,
            f"📤 Одиночная загрузка файла\n\n"
            f"📖 Предмет: {subject['name']}\n\n"
            "Выберите тип файла:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_select_upload_type(self, query, upload_type: str, context: ContextTypes.DEFAULT_TYPE):
        """Обработка выбора типа файла для загрузки"""
        context.user_data['upload_type'] = upload_type
        context.user_data['state'] = 'single_upload_number'
        
        subject_id = context.user_data.get('upload_subject_id')
        subject = self.db.get_subject(subject_id)
        
        type_text = "лекцию" if upload_type == "lecture" else "практическую работу"
        
        await self.edit_message_with_cleanup(
            query, context,
            f"📤 Одиночная загрузка файла\n\n"
            f"📖 Предмет: {subject['name']}\n"
            f"📄 Тип: {type_text}\n\n"
            "Введите номер (например, 1 для Лекции 1):",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад к выбору типа", callback_data=f"upload_subject_{subject_id}")],
                [InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")]
            ])
        )

    async def handle_upload_number(self, update: Update, context: ContextTypes.DEFAULT_TYPE, number_text: str):
        """Обработка ввода номера для загрузки"""
        try:
            number = int(number_text.strip())
            if number <= 0:
                raise ValueError("Номер должен быть положительным числом")
        except ValueError:
            await self.send_message_with_cleanup(update, context, "❌ Введите корректный номер (положительное число)")
            return
        
        context.user_data['upload_number'] = number
        context.user_data['state'] = 'single_upload_file'
        
        subject_id = context.user_data.get('upload_subject_id')
        upload_type = context.user_data.get('upload_type')
        subject = self.db.get_subject(subject_id)
        
        type_text = "лекцию" if upload_type == "lecture" else "практическую работу"
        
        await self.send_message_with_cleanup(
            update, context,
            f"📤 Одиночная загрузка файла\n\n"
            f"📖 Предмет: {subject['name']}\n"
            f"📄 Тип: {type_text}\n"
            f"🔢 Номер: {number}\n\n"
            "Теперь отправьте файл:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")]
            ])
        )

    async def save_single_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сохранение одиночного файла"""
        if not update.message.document:
            await self.send_message_with_cleanup(update, context, "❌ Пожалуйста, отправьте файл.")
            return
        
        subject_id = context.user_data.get('upload_subject_id')
        upload_type = context.user_data.get('upload_type')
        number = context.user_data.get('upload_number')
        
        if not all([subject_id, upload_type, number]):
            await self.send_message_with_cleanup(update, context, "❌ Ошибка: не все параметры загрузки установлены")
            context.user_data.clear()
            return
        
        file = await update.message.document.get_file()
        filename = update.message.document.file_name
        
        try:
            # Создаем директории если не существуют
            base_dir = "lectures" if upload_type == "lecture" else "practices"
            os.makedirs(base_dir, exist_ok=True)
            
            file_path = os.path.join(base_dir, filename)
            await file.download_to_drive(file_path)
            
            subject = self.db.get_subject(subject_id)
            
            # Сохраняем в базу данных
            if upload_type == "lecture":
                lecture_id = self.db.add_lecture(subject_id, number, file_path)
                success = lecture_id is not None
                type_name = "лекция"
            else:
                practice_id = self.db.add_practice(subject_id, number, file_path)
                success = practice_id is not None
                type_name = "практическая работа"
            
            if success:
                await self.send_message_with_cleanup(
                    update, context,
                    f"✅ {type_name.capitalize()} №{number} успешно загружена!\n\n"
                    f"📖 Предмет: {subject['name']}\n"
                    f"📄 Файл: {filename}",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("📤 Загрузить еще файл", callback_data="upload_file")],
                        [InlineKeyboardButton("📚 Просмотреть предметы", callback_data="subjects")],
                        [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                    ])
                )
            else:
                await self.send_message_with_cleanup(
                    update, context,
                    f"❌ Не удалось загрузить {type_name}",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔄 Попробовать снова", callback_data="upload_file")],
                        [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                    ])
                )
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            await self.send_message_with_cleanup(
                update, context,
                f"❌ Ошибка при загрузке файла: {str(e)}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
        
        context.user_data.clear()

    async def start_upload_schedule(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Начать загрузку расписания"""
        if query.from_user.id not in ADMIN_IDS:
            await query.answer("❌ У вас нет доступа", show_alert=True)
            return
        
        context.user_data.clear()
        context.user_data['state'] = 'uploading_schedule'
        
        await self.edit_message_with_cleanup(
            query, context,
            "📤 Загрузка расписания\n\n"
            "Отправьте файл расписания (Excel, PDF, Word, изображение):\n\n"
            "📝 Поддерживаемые форматы:\n"
            "• Excel (.xlsx, .xls)\n"
            "• PDF (.pdf)\n"
            "• Word (.doc, .docx)\n"
            "• Изображения (.jpg, .png, etc.)\n\n"
            "❌ Для отмены используйте /cancel"
        )

    async def handle_schedule_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE, title: str):
        """Обработка загрузки расписания с названием"""
        if not title.strip():
            await self.send_message_with_cleanup(update, context, "❌ Пожалуйста, укажите название расписания.")
            return
        
        context.user_data['schedule_title'] = title.strip()
        
        await self.send_message_with_cleanup(
            update, context,
            f"📝 Название расписания: {title}\n\n"
            "Теперь отправьте файл расписания."
        )

    

    async def update_code_from_github_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Обновить код из GitHub через callback"""
        await self.edit_message_with_cleanup(
            query, context,
            "🔄 Обновление кода из GitHub...\n\n"
            "Это может занять несколько минут. Пожалуйста, подождите."
        )
        
        # Исправленный вызов - теперь это асинхронный метод
        success, message = await self.code_manager.update_code_from_github()
        
        await self.edit_message_with_cleanup(
            query, context,
            message,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Перезапустить", callback_data="restart_bot")] if success else [],
                [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )





    

async def save_schedule_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сохранение загруженного расписания"""
    if not update.message.document:
        await self.send_message_with_cleanup(update, context, "❌ Пожалуйста, отправьте файл расписания.")
        return
    
    file = await update.message.document.get_file()
    filename = update.message.document.file_name
    
    # Получаем название расписания
    schedule_title = context.user_data.get('schedule_title')
    if not schedule_title:
        # Если название не было задано, используем имя файла
        schedule_title = os.path.splitext(filename)[0]
    
    try:
        # Создаем директорию для расписаний если не существует
        schedule_dir = "schedules"
        os.makedirs(schedule_dir, exist_ok=True)
        
        file_path = os.path.join(schedule_dir, filename)
        await file.download_to_drive(file_path)
        
        # Сохраняем в базу данных
        schedule_id = self.db.add_schedule(schedule_title, file_path)
        
        context.user_data.clear()
        
        await self.send_message_with_cleanup(
            update, context,
            f"✅ Расписание '{schedule_title}' успешно загружено!\n\n"
            f"📁 Файл: {filename}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📅 Просмотреть расписание", callback_data="view_schedule")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении расписания: {e}")
        await self.send_message_with_cleanup(
            update, context,
            f"❌ Ошибка при сохранении расписания: {str(e)}"
        )

async def show_schedule_list(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Показать список расписаний (админ)"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
    
    schedules = self.db.get_all_schedule()
    
    if not schedules:
        await self.edit_message_with_cleanup(
            query, context,
            "📅 Расписание\n\nПока нет загруженных расписаний.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📤 Загрузить расписание", callback_data="upload_schedule")],
                [InlineKeyboardButton("🔙 Назад", callback_data="manage_schedule")]
            ])
        )
        return
    
    keyboard = []
    for schedule in schedules:
        keyboard.append([
            InlineKeyboardButton(f"📥 {schedule['title']}", callback_data=f"download_schedule_{schedule['id']}"),
            InlineKeyboardButton("🗑️", callback_data=f"delete_schedule_{schedule['id']}")
        ])
    
    keyboard.append([InlineKeyboardButton("📤 Загрузить новое расписание", callback_data="upload_schedule")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="manage_schedule")])
    
    await self.edit_message_with_cleanup(
        query, context,
        "📅 Загруженные расписания:\n\n"
        "📥 - скачать\n🗑️ - удалить",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def send_schedule_file(self, query, schedule_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Отправить файл расписания"""
    try:
        schedule = self.db.get_schedule(schedule_id)
        
        if not schedule:
            await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Расписание не найдено")
            return
        
        file_path = schedule['file_path']
        
        if not os.path.exists(file_path):
            await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Файл не найден на сервере")
            return
        
        caption = f"📅 {schedule['title']}"
        if schedule.get('description'):
            caption += f"\n\n{schedule['description']}"
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        with open(file_path, 'rb') as file:
            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                await query.message.reply_photo(
                    photo=file,
                    caption=caption
                )
            elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                await query.message.reply_video(
                    video=file,
                    caption=caption
                )
            else:
                await query.message.reply_document(
                    document=file,
                    caption=caption
                )
        
        await self.send_message_with_cleanup(
            Update(update_id=0, callback_query=query), context,
            "Файл расписания отправлен!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📅 Еще расписания", callback_data="schedule")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
            ])
        )
        
    except Exception as e:
        logger.error(f"Ошибка при отправке расписания: {e}")
        await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Ошибка при отправке файла")

async def delete_schedule(self, query, schedule_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Удалить расписание"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
    
    try:
        schedule = self.db.get_schedule(schedule_id)
        
        if not schedule:
            await self.edit_message_with_cleanup(query, context, "❌ Расписание не найдено")
            return
        
        # Удаляем файл с диска
        file_path = schedule['file_path']
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Удаляем запись из базы данных
        self.db.delete_schedule(schedule_id)
        
        await self.edit_message_with_cleanup(
            query, context,
            f"✅ Расписание '{schedule['title']}' успешно удалено!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📅 Управление расписанием", callback_data="manage_schedule")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
        
    except Exception as e:
        logger.error(f"Ошибка при удалении расписания: {e}")
        await self.edit_message_with_cleanup(
            query, context,
            f"❌ Ошибка при удалении расписания: {str(e)}"
        )

async def show_useful_info(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Показать полезную информацию"""
    useful_content = self.db.get_all_useful_content()
    
    if not useful_content:
        await self.edit_message_with_cleanup(
            query, context,
            "ℹ️ Полезная информация\n\nПока нет доступной полезной информации.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
            ])
        )
        return
    
    keyboard = []
    for content in useful_content:
        folder_name = content.get('folder_name', 'Без папки')
        keyboard.append([
            InlineKeyboardButton(
                f"📄 {content['title']} ({folder_name})", 
                callback_data=f"download_useful_{content['id']}"
            )
        ])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
    
    await self.edit_message_with_cleanup(
        query, context,
        "ℹ️ Полезная информация\nВыберите файл для скачивания:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def manage_useful_info(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Управление полезной информацией (админ)"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
    
    useful_content = self.db.get_all_useful_content()
    folders = self.db.get_all_useful_folders()
    
    keyboard = [
        [InlineKeyboardButton("📤 Загрузить информацию", callback_data="upload_useful_info")],
        [InlineKeyboardButton("📋 Просмотреть информацию", callback_data="view_useful_info")]
    ]
    
    if useful_content:
        for content in useful_content:
            keyboard.append([
                InlineKeyboardButton(f"🗑️ Удалить: {content['title']}", callback_data=f"delete_useful_{content['id']}")
            ])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_panel")])
    
    await self.edit_message_with_cleanup(
        query, context,
        "📦 Управление полезной информацией\n\n"
        f"📊 Загружено файлов: {len(useful_content)}\n"
        f"📁 Папок: {len(folders)}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def start_upload_useful_info(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Начать загрузку полезной информации"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
    
    context.user_data.clear()
    context.user_data['state'] = 'uploading_useful_info'
    
    await self.edit_message_with_cleanup(
        query, context,
        "📤 Загрузка полезной информации\n\n"
        "Введите название для этого контента:",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Отмена", callback_data="manage_useful_info")]
        ])
    )

async def handle_useful_info_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE, title: str):
    """Обработка загрузки полезной информации с названием"""
    if not title.strip():
        await self.send_message_with_cleanup(update, context, "❌ Пожалуйста, укажите название.")
        return
    
    context.user_data['useful_title'] = title.strip()
    
    await self.send_message_with_cleanup(
        update, context,
        f"📝 Название: {title}\n\n"
        "Теперь отправьте файл."
    )

async def save_useful_info_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сохранение загруженной полезной информации"""
    if not update.message.document:
        await self.send_message_with_cleanup(update, context, "❌ Пожалуйста, отправьте файл.")
        return
    
    file = await update.message.document.get_file()
    filename = update.message.document.file_name
    
    # Получаем название
    useful_title = context.user_data.get('useful_title')
    if not useful_title:
        # Если название не было задано, используем имя файла
        useful_title = os.path.splitext(filename)[0]
    
    try:
        # Создаем директорию для полезной информации если не существует
        useful_dir = "useful_info"
        os.makedirs(useful_dir, exist_ok=True)
        
        file_path = os.path.join(useful_dir, filename)
        await file.download_to_drive(file_path)
        
        # Определяем тип файла
        file_extension = os.path.splitext(filename)[1].lower()
        content_type = "document"
        if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            content_type = "image"
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            content_type = "video"
        
        # Сохраняем в базу данных
        content_id = self.db.add_useful_content(useful_title, file_path, content_type)
        
        context.user_data.clear()
        
        await self.send_message_with_cleanup(
            update, context,
            f"✅ Полезная информация '{useful_title}' успешно загружена!\n\n"
            f"📁 Файл: {filename}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📋 Просмотреть информацию", callback_data="view_useful_info")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении полезной информации: {e}")
        await self.send_message_with_cleanup(
            update, context,
            f"❌ Ошибка при сохранении: {str(e)}"
        )

async def show_useful_info_list(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Показать список полезной информации (админ)"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
    
    useful_content = self.db.get_all_useful_content()
    
    if not useful_content:
        await self.edit_message_with_cleanup(
            query, context,
            "📦 Полезная информация\n\nПока нет загруженной информации.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📤 Загрузить информацию", callback_data="upload_useful_info")],
                [InlineKeyboardButton("🔙 Назад", callback_data="manage_useful_info")]
            ])
        )
        return
    
    keyboard = []
    for content in useful_content:
        keyboard.append([
            InlineKeyboardButton(f"📥 {content['title']}", callback_data=f"download_useful_{content['id']}"),
            InlineKeyboardButton("🗑️", callback_data=f"delete_useful_{content['id']}")
        ])
    
    keyboard.append([InlineKeyboardButton("📤 Загрузить новую информацию", callback_data="upload_useful_info")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="manage_useful_info")])
    
    await self.edit_message_with_cleanup(
        query, context,
        "📦 Загруженная полезная информация:\n\n"
        "📥 - скачать\n🗑️ - удалить",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def send_useful_file(self, query, content_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Отправить файл полезной информации"""
    try:
        content = self.db.get_useful_content(content_id)
        
        if not content:
            await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Контент не найден")
            return
        
        file_path = content['file_path']
        
        if not os.path.exists(file_path):
            await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Файл не найден на сервере")
            return
        
        caption = f"📄 {content['title']}"
        if content.get('folder_name'):
            caption += f" ({content['folder_name']})"
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        with open(file_path, 'rb') as file:
            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                await query.message.reply_photo(
                    photo=file,
                    caption=caption
                )
            elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                await query.message.reply_video(
                    video=file,
                    caption=caption
                )
            else:
                await query.message.reply_document(
                    document=file,
                    caption=caption
                )
        
        await self.send_message_with_cleanup(
            Update(update_id=0, callback_query=query), context,
            "Файл отправлен!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("ℹ️ Еще информация", callback_data="useful_info")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")]
            ])
        )
        
    except Exception as e:
        logger.error(f"Ошибка при отправке полезной информации: {e}")
        await self.send_message_with_cleanup(Update(update_id=0, callback_query=query), context, "❌ Ошибка при отправке файла")

async def delete_useful_content(self, query, content_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Удалить полезную информацию"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
    
    try:
        content = self.db.get_useful_content(content_id)
        
        if not content:
            await self.edit_message_with_cleanup(query, context, "❌ Контент не найден")
            return
        
        # Удаляем файл с диска
        file_path = content['file_path']
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Удаляем запись из базы данных
        self.db.delete_useful_content(content_id)
        
        await self.edit_message_with_cleanup(
            query, context,
            f"✅ Контент '{content['title']}' успешно удален!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📦 Управление информацией", callback_data="manage_useful_info")],
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
        
    except Exception as e:
        logger.error(f"Ошибка при удалении контента: {e}")
        await self.edit_message_with_cleanup(
            query, context,
            f"❌ Ошибка при удалении контента: {str(e)}"
        )

async def show_admin_panel(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Показать админ-панель из команды"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
        
    keyboard = [
        [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
        [InlineKeyboardButton("📤 Одиночная загрузка", callback_data="upload_file")],
        [InlineKeyboardButton("📚 Массовая загрузка", callback_data="mass_upload")],  # ← ЭТА КНОПКА
        [InlineKeyboardButton("🗑️ Удаление файлов", callback_data="delete_files")],   # ← И ЭТА
        [InlineKeyboardButton("➕ Добавить предмет", callback_data="add_subject")],
        [InlineKeyboardButton("👨‍🏫 Добавить преподавателя", callback_data="add_teacher")],
        [InlineKeyboardButton("📅 Управление расписанием", callback_data="manage_schedule")],
        [InlineKeyboardButton("📦 Управление полезной инфо", callback_data="manage_useful_info")],
        [InlineKeyboardButton("📋 Просмотр логов", callback_data="view_logs")],
        [InlineKeyboardButton("🤖 Статистика ИИ", callback_data="ai_stats")],
        [InlineKeyboardButton("🔧 Управление кодом", callback_data="code_manager")],
        [InlineKeyboardButton("🔙 Главное меню", callback_data="back_to_menu")]
    ]
    
    await self.edit_message_with_cleanup(
        query, context,
        "⚙️ Админ-панель\nВыберите действие:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def show_support(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Показать информацию о техподдержке и перенаправить в канал"""
    text = (
        "📞 Техническая поддержка\n\n"
        f"Если у вас возникли проблемы или вопросы, "
        f"обратитесь в нашу группу поддержки.\n\n"
        "Мы поможем вам решить любые проблемы!"
    )
    
    keyboard = [
        [InlineKeyboardButton("📞 Перейти в поддержку", url=f"https://t.me/{SUPPORT_GROUP_ID.replace('@', '')}")],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
    ]
    
    await self.edit_message_with_cleanup(
        query, context,
        text,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )





# =============================================================================
# ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ ДЛЯ ОБРАБОТКИ КНОПОК
# =============================================================================



async def handle_add_subject(self, update: Update, context: ContextTypes.DEFAULT_TYPE, subject_name: str):
    """Обработка добавления предмета"""
    if not subject_name.strip():
        await self.send_message_with_cleanup(update, context, "❌ Название предмета не может быть пустым")
        return
    
    try:
        subject_id = self.db.add_subject(subject_name.strip())
        
        if subject_id:
            await self.send_message_with_cleanup(
                update, context,
                f"✅ Предмет '{subject_name}' успешно добавлен!",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Добавить еще предмет", callback_data="add_subject")],
                    [InlineKeyboardButton("📚 Просмотреть предметы", callback_data="subjects")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
        else:
            await self.send_message_with_cleanup(
                update, context,
                f"❌ Не удалось добавить предмет '{subject_name}'",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Попробовать снова", callback_data="add_subject")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
    
    except Exception as e:
        logger.error(f"Ошибка при добавлении предмета: {e}")
        await self.send_message_with_cleanup(
            update, context,
            f"❌ Ошибка при добавлении предмета: {str(e)}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
    
    context.user_data.clear()


async def handle_select_subject_for_teacher(self, query, subject_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора предмета для преподавателя"""
    context.user_data['teacher_subject_id'] = subject_id
    context.user_data['state'] = 'adding_teacher_name'
    
    subject = self.db.get_subject(subject_id)
    
    await self.edit_message_with_cleanup(
        query, context,
        f"👨‍🏫 Добавление преподавателя\n\n"
        f"📖 Предмет: {subject['name']}\n\n"
        "Введите ФИО преподавателя:",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад к выбору предмета", callback_data="add_teacher")],
            [InlineKeyboardButton("❌ Отмена", callback_data="admin_panel")]
        ])
    )

async def handle_add_teacher(self, update: Update, context: ContextTypes.DEFAULT_TYPE, teacher_name: str):
    """Обработка добавления преподавателя"""
    if not teacher_name.strip():
        await self.send_message_with_cleanup(update, context, "❌ ФИО преподавателя не может быть пустым")
        return
    
    subject_id = context.user_data.get('teacher_subject_id')
    if not subject_id:
        await self.send_message_with_cleanup(update, context, "❌ Ошибка: предмет не выбран")
        context.user_data.clear()
        return
    
    try:
        teacher_id = self.db.add_teacher(teacher_name.strip(), subject_id)
        subject = self.db.get_subject(subject_id)
        
        if teacher_id:
            await self.send_message_with_cleanup(
                update, context,
                f"✅ Преподаватель '{teacher_name}' успешно добавлен!\n"
                f"📖 Предмет: {subject['name']}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("👨‍🏫 Добавить еще преподавателя", callback_data="add_teacher")],
                    [InlineKeyboardButton("📚 Просмотреть предметы", callback_data="subjects")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
        else:
            await self.send_message_with_cleanup(
                update, context,
                f"❌ Не удалось добавить преподавателя '{teacher_name}'",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Попробовать снова", callback_data="add_teacher")],
                    [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
                ])
            )
    
    except Exception as e:
        logger.error(f"Ошибка при добавлении преподавателя: {e}")
        await self.send_message_with_cleanup(
            update, context,
            f"❌ Ошибка при добавлении преподавателя: {str(e)}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
            ])
        )
    
    context.user_data.clear()

async def diagnose_training(self, query, context: ContextTypes.DEFAULT_TYPE):
    """Диагностика обучения датасетов"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer("❌ У вас нет доступа", show_alert=True)
        return
    
    # Получаем список датасетов
    datasets = self.ai_assistant.get_datasets_info()
    
    if not datasets:
        await self.edit_message_with_cleanup(
            query, context,
            "📚 Диагностика обучения\n\n❌ Нет доступных датасетов для диагностики."
        )
        return
    
    diagnostic_results = []
    
    for dataset in datasets[:3]:  # Проверяем первые 3 датасета
        dataset_name = dataset['filename']
        
        try:
            # 1. Проверяем существование файла
            filepath = os.path.join("training_datasets", dataset_name)
            if not os.path.exists(filepath):
                diagnostic_results.append(f"❌ {dataset_name}: Файл не найден")
                continue
            
            # 2. Проверяем размер файла
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                diagnostic_results.append(f"❌ {dataset_name}: Файл пустой ({file_size} байт)")
                continue
            
            # 3. Пробуем загрузить датасет
            X, y = self.ai_assistant.self_learning_ai.dataset_trainer.load_dataset(dataset_name)
            
            if len(X) == 0:
                diagnostic_results.append(f"❌ {dataset_name}: Не удалось извлечь данные (X пустой)")
                continue
            
            if len(y) == 0:
                diagnostic_results.append(f"❌ {dataset_name}: Не удалось извлечь метки (y пустой)")
                continue
            
            # 4. Проверяем размерности
            dim_info = f"X: {X.shape}, y: {y.shape}, классы: {len(np.unique(y))}"
            
            # 5. Пробуем небольшое обучение (2 примера, 1 эпоха)
            try:
                test_X = X[:2]  # Берем всего 2 примера для теста
                test_y = y[:2]
                
                losses = self.ai_assistant.self_learning_ai.learn_from_data(
                    test_X, test_y, epochs=1, batch_size=2
                )
                
                if losses and len(losses) > 0:
                    diagnostic_results.append(f"✅ {dataset_name}: ОБУЧЕНИЕ РАБОТАЕТ! {dim_info}")
                else:
                    diagnostic_results.append(f"❌ {dataset_name}: Обучение не запускается {dim_info}")
                    
            except Exception as e:
                diagnostic_results.append(f"❌ {dataset_name}: Ошибка обучения: {str(e)[:100]}...")
                
        except Exception as e:
            diagnostic_results.append(f"❌ {dataset_name}: Общая ошибка: {str(e)[:100]}...")
    
    # Формируем итоговое сообщение
    result_text = "🔍 Диагностика обучения датасетов\n\n"
    result_text += "\n".join(diagnostic_results)
    
    # Добавляем рекомендации
    result_text += "\n\n💡 Рекомендации:\n"
    result_text += "• ✅ - датасет готов к обучению\n"
    result_text += "• ❌ - требуется исправление\n"
    result_text += "• Проверьте формат данных в проблемных датасетах"
    
    await self.edit_message_with_cleanup(
        query, context,
        result_text,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Запустить обучение", callback_data="train_dataset")],
            [InlineKeyboardButton("📚 Управление датасетами", callback_data="manage_datasets")],
            [InlineKeyboardButton("⚙️ Админ-панель", callback_data="admin_panel")]
        ])
    )

def main():
    """Функция для запуска улучшенного бота"""
    attempt = 0
    max_attempts = 5
        
    while attempt < max_attempts:
        try:
            logger.info(f"Попытка запуска улучшенного бота №{attempt + 1}")
            bot = EnhancedLectureBot()
            bot.run_bot()
        except KeyboardInterrupt:
            logger.info("Бот остановлен пользователем")
            break
        except Exception as e:
            logger.error(f"Ошибка при запуске улучшенного бота: {e}")
            attempt += 1
            if attempt < max_attempts:
                logger.info("Перезапуск через 10 секунд...")
                time.sleep(10)
            else:
                logger.error("Достигнут лимит перезапусков.")
                sys.exit(1)

if __name__ == '__main__':
    main()
