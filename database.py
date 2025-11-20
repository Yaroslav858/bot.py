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
        conn.row_factory = sqlite3.Row  # Для доступа к колонкам по имени
        return conn

    def init_database(self):
        """Инициализация базы данных с поддержкой преподавателей"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Включаем поддержку внешних ключей
        cursor.execute("PRAGMA foreign_keys = ON")
        
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

    '''def get_all_useful_content(self) -> List[Dict[str, Any]]:
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
            conn.close()'''

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
