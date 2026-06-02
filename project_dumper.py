import os
import argparse
import fnmatch
from pathlib import Path

# Базовые исключения (предохранитель).
# Отрабатывают всегда, даже если этих папок/файлов нет в .gitignore
IGNORE_DIRS = {'.git', '.idea', '.vscode', '__pycache__'}

IGNORE_EXTENSIONS = {
    # Изображения и медиа
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.mp3', '.mp4',
    # Архивы и бинарники
    '.zip', '.tar', '.gz', '.rar', '.7z', '.exe', '.dll', '.so', '.dylib', '.bin',
    # Специфичные форматы
    '.pdf', '.pyc', '.pyd', '.sqlite3', '.db'
}

def load_gitignore(target_dir: Path) -> list:
    """Читает файл .gitignore и возвращает список активных правил."""
    patterns = []
    gitignore_path = target_dir / '.gitignore'
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Игнорируем пустые строки и комментарии
                if line and not line.startswith('#'):
                    patterns.append(line)
    return patterns

def should_ignore(item_path: Path, target_dir: Path, gitignore_patterns: list) -> bool:
    """
    Проверяет, нужно ли пропустить файл/папку. 
    Учитывает базовые исключения и правила из .gitignore.
    """
    # 1. Проверка по базовым (жестким) спискам
    if item_path.is_dir() and item_path.name in IGNORE_DIRS:
        return True
    if item_path.is_file() and item_path.suffix.lower() in IGNORE_EXTENSIONS:
        return True

    # 2. Проверка по правилам .gitignore
    if not gitignore_patterns:
        return False

    # Получаем относительный путь от корня проекта (используем / для унификации)
    rel_path = item_path.relative_to(target_dir).as_posix()
    name = item_path.name

    for pattern in gitignore_patterns:
        # Очищаем паттерн от слешей для упрощенного матчинга
        clean_pattern = pattern.strip('/')
        
        # Если паттерн не содержит путей (например, '*.log' или 'node_modules')
        if '/' not in clean_pattern:
            if fnmatch.fnmatch(name, clean_pattern):
                return True
        else:
            # Если паттерн специфичен для пути (например, 'build/outputs')
            if fnmatch.fnmatch(rel_path, clean_pattern) or rel_path.startswith(clean_pattern + '/'):
                return True

    return False

def generate_tree(dir_path: Path, target_dir: Path, gitignore_patterns: list, prefix: str = "") -> str:
    """Рекурсивно строит дерево директорий с учетом исключений."""
    tree_str = ""
    try:
        # Получаем всё содержимое и фильтруем на лету
        entries = sorted(dir_path.iterdir(), key=lambda e: (e.is_file(), e.name))
        valid_entries = [e for e in entries if not should_ignore(e, target_dir, gitignore_patterns)]
        
        entries_count = len(valid_entries)
        for index, entry in enumerate(valid_entries):
            connector = "└── " if index == entries_count - 1 else "├── "
            tree_str += f"{prefix}{connector}{entry.name}\n"
            
            if entry.is_dir():
                extension = "    " if index == entries_count - 1 else "│   "
                tree_str += generate_tree(entry, target_dir, gitignore_patterns, prefix + extension)
    except PermissionError:
        tree_str += f"{prefix}└── [Отказано в доступе]\n"
        
    return tree_str

def write_project_dump(target_dir: str, output_file: str):
    target_path = Path(target_dir).resolve()
    
    # Загружаем правила
    gitignore_patterns = load_gitignore(target_path)
    if gitignore_patterns:
        print(f"Обнаружен .gitignore! Загружено правил: {len(gitignore_patterns)}")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"=== СТРУКТУРА ПРОЕКТА: {target_path.name} ===\n\n")
        out.write(f"{target_path.name}/\n")
        
        # Строим дерево
        out.write(generate_tree(target_path, target_path, gitignore_patterns))
        
        out.write("\n\n=== СОДЕРЖИМОЕ ФАЙЛОВ ===\n\n")
        
        # Обходим файлы для чтения их содержимого
        for root, dirs, files in os.walk(target_path):
            current_root = Path(root)
            
            # Модифицируем список dirs "на месте", чтобы os.walk не заходил в игнорируемые папки
            dirs[:] = [d for d in dirs if not should_ignore(current_root / d, target_path, gitignore_patterns)]
            
            for file in sorted(files):
                file_path = current_root / file
                
                # Проверяем сам файл
                if should_ignore(file_path, target_path, gitignore_patterns):
                    continue
                
                relative_path = file_path.relative_to(target_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    out.write(f"--- Файл: {relative_path} ---\n")
                    out.write("```" + (file_path.suffix.lstrip('.') or 'text') + "\n")
                    out.write(content)
                    if not content.endswith('\n'):
                        out.write("\n")
                    out.write("```\n\n")
                    
                except UnicodeDecodeError:
                    out.write(f"--- Файл: {relative_path} [ПРОПУЩЕН: Не текстовый формат] ---\n\n")
                except Exception as e:
                    out.write(f"--- Файл: {relative_path} [ОШИБКА ЧТЕНИЯ: {e}] ---\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дамп структуры и кода проекта с поддержкой .gitignore")
    parser.add_argument("target_folder", nargs="?", default=".", help="Путь к папке проекта (по умолчанию текущая)")
    parser.add_argument("-o", "--output", default="project_dump.txt", help="Имя выходного файла")
    
    args = parser.parse_args()
    
    print(f"Анализируем папку: {Path(args.target_folder).resolve()}")
    write_project_dump(args.target_folder, args.output)
    print(f"Готово! Результат сохранен в файл: {args.output}")