import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import webbrowser
import glob

# Создаем папку для визуализаций
VISUALIZATION_FOLDER = 'visualization'
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Устанавливаем параметры для корректного отображения графиков
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("=" * 60)
print("АНАЛИЗ ДАННЫХ ОБ ИГРАХ")
print("=" * 60)
print(f"Все файлы будут сохранены в папку: {VISUALIZATION_FOLDER}")

# Загрузка данных
try:
    print("Загрузка данных из Games.csv...")
    df = pd.read_csv('Games.csv')
    print(f"✓ Данные успешно загружены: {len(df)} строк, {len(df.columns)} колонок")
except FileNotFoundError:
    print("✗ Ошибка: Файл 'Games.csv' не найден!")
    print(f"  Текущая директория: {os.getcwd()}")
    exit()

print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о датафрейме:")
print(df.info())
print("\nОписательная статистика по Score:")
print(df['Score'].describe())

# Предобработка данных
print("\nПредобработка данных...")

# Преобразуем Score в числовой тип
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# Разделяем строки с несколькими платформами
df['Console'] = df['Console'].astype(str)
df['Console'] = df['Console'].str.split(',')
df = df.explode('Console')
df['Console'] = df['Console'].str.strip()
df['Console'] = df['Console'].replace(['', 'nan', 'NaN', 'None'], 'Unknown')

# Удаляем пропуски
initial_rows = len(df)
df = df.dropna(subset=['Score'])
print(f"✓ Удалено строк с пропусками: {initial_rows - len(df)}")

# Создаем категории оценок
def categorize_score(score):
    if score >= 9:
        return 'Excellent (9-10)'
    elif score >= 8:
        return 'Great (8-8.9)'
    elif score >= 7:
        return 'Good (7-7.9)'
    elif score >= 6:
        return 'Fair (6-6.9)'
    elif score >= 5:
        return 'Mediocre (5-5.9)'
    elif score >= 4:
        return 'Poor (4-4.9)'
    elif score >= 3:
        return 'Bad (3-3.9)'
    elif score >= 2:
        return 'Terrible (2-2.9)'
    else:
        return 'Abysmal (0-1.9)'

df['Score_Category'] = df['Score'].apply(categorize_score)

# 1. СТАТИЧЕСКИЕ ГРАФИКИ
print("\n" + "=" * 60)
print("СОЗДАНИЕ СТАТИЧЕСКИХ ГРАФИКОВ")
print("=" * 60)

try:
    # 1.1 Основные графики
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Распределение оценок
    ax1 = axes[0, 0]
    ax1.hist(df['Score'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_title('Распределение оценок игр', fontsize=12)
    ax1.set_xlabel('Оценка')
    ax1.set_ylabel('Количество игр')
    mean_score = df['Score'].mean()
    ax1.axvline(mean_score, color='red', linestyle='--', linewidth=1, 
                label=f'Среднее: {mean_score:.2f}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Топ-10 платформ
    ax2 = axes[0, 1]
    top_consoles = df['Console'].value_counts().head(10)
    ax2.bar(range(len(top_consoles)), top_consoles.values, color='lightgreen', alpha=0.8)
    ax2.set_title('Топ-10 платформ по количеству игр', fontsize=12)
    ax2.set_xlabel('Платформа')
    ax2.set_ylabel('Количество игр')
    ax2.set_xticks(range(len(top_consoles)))
    ax2.set_xticklabels(top_consoles.index, rotation=45, ha='right', fontsize=9)
    
    # Средняя оценка по платформам
    ax3 = axes[0, 2]
    platform_avg = df.groupby('Console')['Score'].mean().sort_values(ascending=False).head(15)
    ax3.bar(range(len(platform_avg)), platform_avg.values, color='orange', alpha=0.8)
    ax3.set_title('Топ-15 платформ по средней оценке', fontsize=12)
    ax3.set_xlabel('Платформа')
    ax3.set_ylabel('Средняя оценка')
    ax3.set_xticks(range(len(platform_avg)))
    ax3.set_xticklabels(platform_avg.index, rotation=45, ha='right', fontsize=8)
    ax3.axhline(y=mean_score, color='red', linestyle='--', linewidth=1)
    
    # Категории оценок
    ax4 = axes[1, 0]
    score_cats = df['Score_Category'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(score_cats)))
    ax4.bar(range(len(score_cats)), score_cats.values, color=colors)
    ax4.set_title('Распределение по категориям оценок', fontsize=12)
    ax4.set_xlabel('Категория')
    ax4.set_ylabel('Количество игр')
    ax4.set_xticks(range(len(score_cats)))
    ax4.set_xticklabels(score_cats.index, rotation=45, ha='right', fontsize=8)
    
    # Boxplot для топ-5 платформ
    ax5 = axes[1, 1]
    top_5_consoles = df['Console'].value_counts().head(5).index
    box_data = []
    labels = []
    for console in top_5_consoles:
        scores = df[df['Console'] == console]['Score'].dropna()
        if len(scores) > 0:
            box_data.append(scores)
            labels.append(console)
    
    bp = ax5.boxplot(box_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax5.set_title('Распределение оценок (топ-5 платформ)', fontsize=12)
    ax5.set_xlabel('Платформа')
    ax5.set_ylabel('Оценка')
    ax5.tick_params(axis='x', rotation=45, labelsize=9)
    
    # Плотность распределения
    ax6 = axes[1, 2]
    sns.kdeplot(df['Score'], fill=True, alpha=0.5, color='purple', ax=ax6)
    ax6.set_title('Плотность распределения оценок', fontsize=12)
    ax6.set_xlabel('Оценка')
    ax6.set_ylabel('Плотность')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Анализ оценок видеоигр', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Сохраняем в папку visualization
    png_path1 = os.path.join(VISUALIZATION_FOLDER, 'games_analysis.png')
    try:
        plt.savefig(png_path1, dpi=100, bbox_inches='tight')
        print(f"✓ График сохранен как '{png_path1}'")
    except Exception as e:
        print(f"⚠ Не удалось сохранить PNG: {e}")
    
    plt.show()
    
except Exception as e:
    print(f"✗ Ошибка при создании графиков: {e}")
    import traceback
    traceback.print_exc()

# 2. УГЛУБЛЕННЫЙ АНАЛИЗ
print("\n" + "=" * 60)
print("УГЛУБЛЕННЫЙ АНАЛИЗ")
print("=" * 60)

try:
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Violin plot для топ-5 платформ
    top_5_consoles = df['Console'].value_counts().head(5).index
    df_top_5 = df[df['Console'].isin(top_5_consoles)]
    
    sns.violinplot(x='Console', y='Score', data=df_top_5, 
                   palette='Set2', ax=axes2[0])
    axes2[0].set_title('Violin plot оценок по платформам (топ-5)', fontsize=12)
    axes2[0].set_xlabel('Платформа')
    axes2[0].set_ylabel('Оценка')
    axes2[0].tick_params(axis='x', rotation=45)
    
    # Swarm plot (выборка для производительности)
    sample_size = min(200, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    
    sns.swarmplot(x='Score_Category', y='Score', data=sample_df, 
                  size=3, palette='husl', ax=axes2[1])
    axes2[1].set_title(f'Swarm plot (выборка {sample_size} игр)', fontsize=12)
    axes2[1].set_xlabel('Категория оценки')
    axes2[1].set_ylabel('Оценка')
    axes2[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Углубленный анализ', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Сохраняем в папку visualization
    png_path2 = os.path.join(VISUALIZATION_FOLDER, 'advanced_analysis.png')
    try:
        plt.savefig(png_path2, dpi=100, bbox_inches='tight')
        print(f"✓ Расширенный анализ сохранен как '{png_path2}'")
    except:
        print("⚠ Не удалось сохранить расширенный анализ")
    
    plt.show()
    
except Exception as e:
    print(f"✗ Ошибка при углубленном анализе: {e}")

# 3. HTML ВИЗУАЛИЗАЦИЯ С BOKEH (ИСПРАВЛЕННАЯ ВЕРСИЯ)
print("\n" + "=" * 60)
print("СОЗДАНИЕ ИНТЕРАКТИВНЫХ HTML ГРАФИКОВ")
print("=" * 60)

def create_bokeh_html():
    """
    Создает HTML файлы с интерактивными графиками Bokeh
    """
    try:
        # Импортируем необходимые модули Bokeh
        from bokeh.plotting import figure, output_file, save
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.palettes import Category10
        from bokeh.transform import factor_cmap
        
        print("Подготавливаю данные для графиков...")
        
        # 1. Подготовка данных для топ-10 платформ
        top_10_consoles = df['Console'].value_counts().head(10).index.tolist()
        df_top_10 = df[df['Console'].isin(top_10_consoles)]
        
        # Агрегируем данные
        console_stats = df_top_10.groupby('Console').agg(
            avg_score=('Score', 'mean'),
            count=('Score', 'size'),
            min_score=('Score', 'min'),
            max_score=('Score', 'max')
        ).reset_index()
        
        console_stats['avg_score'] = console_stats['avg_score'].round(2)
        console_stats = console_stats.sort_values('avg_score', ascending=False)
        
        print(f"Найдено {len(console_stats)} платформ в топ-10")
        
        # 2. Файл 1: Средняя оценка по платформам
        html_path1 = os.path.join(VISUALIZATION_FOLDER, 'games_interactive.html')
        print(f"\n1. Создаю файл: {html_path1}")
        
        try:
            output_file(html_path1)
            
            source1 = ColumnDataSource(console_stats)
            
            p1 = figure(
                title="Средняя оценка игр по платформам (топ-10)",
                x_range=console_stats['Console'].tolist(),
                width=1000,
                height=500,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                toolbar_location="above"
            )
            
            colors = Category10[10]
            
            # Создаем столбчатую диаграмму
            p1.vbar(
                x='Console',
                top='avg_score',
                width=0.7,
                source=source1,
                line_color='white',
                fill_color=factor_cmap('Console', palette=colors[:len(console_stats)], 
                                      factors=console_stats['Console'].tolist()),
                line_width=1.5
            )
            
            p1.xaxis.major_label_orientation = 45
            p1.xaxis.axis_label = "Платформа"
            p1.yaxis.axis_label = "Средняя оценка"
            p1.y_range.start = 0
            
            # Добавляем подсказки
            hover1 = HoverTool()
            hover1.tooltips = [
                ("Платформа", "@Console"),
                ("Средняя оценка", "@avg_score"),
                ("Количество игр", "@count"),
                ("Мин/Макс", "@min_score / @max_score")
            ]
            p1.add_tools(hover1)
            
            # Сохраняем
            save(p1)
            print(f"   ✅ Файл успешно создан: {os.path.basename(html_path1)}")
            
        except Exception as e:
            print(f"   ❌ Ошибка при создании games_interactive.html: {e}")
            return []
        
        # 3. Файл 2: Количество игр по платформам
        html_path2 = os.path.join(VISUALIZATION_FOLDER, 'games_count.html')
        print(f"\n2. Создаю файл: {html_path2}")
        
        try:
            output_file(html_path2)
            
            source2 = ColumnDataSource(console_stats.sort_values('count', ascending=False))
            
            p2 = figure(
                title="Количество игр по платформам (топ-10)",
                x_range=console_stats.sort_values('count', ascending=False)['Console'].tolist(),
                width=1000,
                height=500,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                toolbar_location="above"
            )
            
            p2.vbar(
                x='Console',
                top='count',
                width=0.7,
                source=source2,
                line_color='white',
                fill_color='#2E7D32',
                alpha=0.7
            )
            
            p2.xaxis.major_label_orientation = 45
            p2.xaxis.axis_label = "Платформа"
            p2.yaxis.axis_label = "Количество игр"
            p2.y_range.start = 0
            
            hover2 = HoverTool()
            hover2.tooltips = [
                ("Платформа", "@Console"),
                ("Количество игр", "@count"),
                ("Средняя оценка", "@avg_score")
            ]
            p2.add_tools(hover2)
            
            save(p2)
            print(f"   ✅ Файл успешно создан: {os.path.basename(html_path2)}")
            
        except Exception as e:
            print(f"   ❌ Ошибка при создании games_count.html: {e}")
        
        # 4. Файл 3: Scatter plot распределения оценок
        html_path3 = os.path.join(VISUALIZATION_FOLDER, 'games_scatter.html')
        print(f"\n3. Создаю файл: {html_path3}")
        
        try:
            output_file(html_path3)
            
            # Берем выборку для scatter plot
            sample_size_scatter = min(500, len(df))
            scatter_sample = df.sample(sample_size_scatter, random_state=42)
            
            # Добавляем немного jitter для лучшей визуализации
            np.random.seed(42)
            scatter_sample = scatter_sample.copy()
            scatter_sample['jitter'] = np.random.uniform(-0.3, 0.3, len(scatter_sample))
            
            source3 = ColumnDataSource(scatter_sample)
            
            p3 = figure(
                title=f"Распределение оценок (выборка {sample_size_scatter} игр)",
                width=1000,
                height=500,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                toolbar_location="above"
            )
            
            # Цвета по категориям
            categories = sorted(scatter_sample['Score_Category'].unique())
            colors_scatter = Category10[len(categories)] if len(categories) <= 10 else Category10[10]
            
            p3.scatter(
                x='Score',
                y='jitter',
                size=10,
                source=source3,
                color=factor_cmap('Score_Category', palette=colors_scatter, 
                                 factors=categories),
                alpha=0.6,
                legend_group='Score_Category'
            )
            
            p3.xaxis.axis_label = "Оценка игры"
            p3.yaxis.axis_label = ""
            p3.yaxis.visible = False
            p3.legend.title = "Категория оценки"
            p3.legend.location = "top_left"
            
            hover3 = HoverTool()
            hover3.tooltips = [
                ("Платформа", "@Console"),
                ("Оценка", "@Score"),
                ("Категория", "@Score_Category")
            ]
            p3.add_tools(hover3)
            
            save(p3)
            print(f"   ✅ Файл успешно создан: {os.path.basename(html_path3)}")
            
        except Exception as e:
            print(f"   ❌ Ошибка при создании games_scatter.html: {e}")
        
        return [html_path1, html_path2, html_path3]
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка в create_bokeh_html: {e}")
        import traceback
        traceback.print_exc()
        return []

# Проверяем и устанавливаем Bokeh при необходимости
def check_and_install_bokeh():
    """
    Проверяет наличие Bokeh и устанавливает его при необходимости
    """
    try:
        import bokeh
        bokeh_version = bokeh.__version__
        print(f"✅ Bokeh установлен (версия {bokeh_version})")
        return True
    except ImportError:
        print("\n❌ Bokeh не установлен. Пытаюсь установить...")
        try:
            import subprocess
            import sys
            
            print("Устанавливаю Bokeh через pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "bokeh", "-q"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Bokeh успешно установлен")
                import bokeh
                return True
            else:
                print(f"❌ Не удалось установить Bokeh: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка при установке Bokeh: {e}")
            return False

# Проверяем Bokeh
bokeh_installed = check_and_install_bokeh()

# Создаем HTML файлы если Bokeh установлен
html_files = []
if bokeh_installed:
    print("\nНачинаю создание интерактивных графиков...")
    html_files = create_bokeh_html()
    
    if html_files:
        print(f"\n✅ Создано {len([f for f in html_files if os.path.exists(f)])} из 3 файлов")
        for file_path in html_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"   - {os.path.basename(file_path)}: {size:,} байт")
            else:
                print(f"   - {os.path.basename(file_path)}: ФАЙЛ НЕ СОЗДАН!")
    else:
        print("\n⚠ Не удалось создать интерактивные графики")
else:
    print("\n⚠ Bokeh не установлен, создаю заглушки...")
    
    # Создаем простой HTML файл для games_interactive.html
    html_path1 = os.path.join(VISUALIZATION_FOLDER, 'games_interactive.html')
    html_content1 = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Средние оценки игр по платформам</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            .info { background: #f0f7ff; padding: 20px; border-radius: 5px; margin: 20px 0; }
            code { background: #eee; padding: 5px 10px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Средние оценки игр по платформам</h1>
            <div class="info">
                <p>⚠️ Для отображения интерактивного графика необходимо установить библиотеку Bokeh.</p>
                <p>Выполните в командной строке:</p>
                <p><code>pip install bokeh</code></p>
                <p>Затем запустите скрипт анализа снова.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(html_path1, 'w', encoding='utf-8') as f:
        f.write(html_content1)
    html_files.append(html_path1)
    print(f"✅ Создан простой HTML файл: '{html_path1}'")

# 4. СОХРАНЕНИЕ ДАННЫХ
try:
    csv_path = os.path.join(VISUALIZATION_FOLDER, 'processed_games.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n✓ Обработанные данные сохранены в '{csv_path}'")
except Exception as e:
    print(f"✗ Ошибка при сохранении данных: {e}")

# 5. СОЗДАНИЕ ОТЧЕТА В HTML
print("\n" + "=" * 60)
print("СОЗДАНИЕ HTML ОТЧЕТА")
print("=" * 60)

try:
    html_report_path = os.path.join(VISUALIZATION_FOLDER, 'report.html')
    
    # Собираем статистику для отчета
    total_games = len(df)
    unique_platforms = df['Console'].nunique()
    avg_score = df['Score'].mean()
    min_score = df['Score'].min()
    max_score = df['Score'].max()
    
    # Топ платформ
    top_platforms_stats = df.groupby('Console').agg(
        games_count=('Score', 'size'),
        avg_score=('Score', 'mean')
    ).sort_values('games_count', ascending=False).head(5)
    
    # Проверяем какие файлы действительно создались
    actual_files = []
    for file in os.listdir(VISUALIZATION_FOLDER):
        if file.endswith(('.html', '.png', '.csv')):
            file_path = os.path.join(VISUALIZATION_FOLDER, file)
            size = os.path.getsize(file_path)
            actual_files.append((file, size))
    
    # Создаем HTML отчет
    html_content = f'''
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет анализа игр</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #444; margin-top: 30px; }}
            .stats {{ background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .stat-item {{ margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #4CAF50; }}
            .files {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
            .file-card {{ background: #e8f5e9; padding: 15px; border-radius: 5px; border: 1px solid #c8e6c9; }}
            .file-card a {{ display: block; color: #2e7d32; text-decoration: none; font-weight: bold; margin: 5px 0; }}
            .file-card a:hover {{ color: #1b5e20; text-decoration: underline; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .images {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
            .image-container {{ flex: 1; min-width: 300px; }}
            .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .file-list {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .file-item {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
            .success {{ color: #4CAF50; font-weight: bold; }}
            .warning {{ color: #ff9800; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Отчет анализа видеоигр</h1>
            
            <div class="stats">
                <h2>📈 Основная статистика</h2>
                <div class="stat-item"><strong>Всего игр:</strong> {total_games:,}</div>
                <div class="stat-item"><strong>Уникальных платформ:</strong> {unique_platforms}</div>
                <div class="stat-item"><strong>Средняя оценка:</strong> {avg_score:.2f}</div>
                <div class="stat-item"><strong>Диапазон оценок:</strong> {min_score} - {max_score}</div>
            </div>
            
            <h2>🎮 Топ-5 платформ по количеству игр</h2>
            <table>
                <tr><th>Платформа</th><th>Количество игр</th><th>Средняя оценка</th></tr>
    '''
    
    # Добавляем строки с топ платформами
    for platform, row in top_platforms_stats.iterrows():
        html_content += f'<tr><td>{platform}</td><td>{row["games_count"]:,}</td><td>{row["avg_score"]:.2f}</td></tr>'
    
    html_content += f'''
            </table>
            
            <h2>📁 Все созданные файлы</h2>
            <div class="file-list">
    '''
    
    # Добавляем список файлов
    for file_name, file_size in actual_files:
        file_ext = file_name.split('.')[-1]
        if file_ext == "html":
            icon = "📊" if "games_" in file_name else "📄"
        elif file_ext == "png":
            icon = "🖼️"
        elif file_ext == "csv":
            icon = "📋"
        else:
            icon = "📁"
        
        status = "✅" if os.path.exists(os.path.join(VISUALIZATION_FOLDER, file_name)) else "❌"
        html_content += f'<div class="file-item">{status} {icon} <a href="{file_name}" target="_blank">{file_name}</a> ({file_size:,} байт)</div>'
    
    html_content += f'''
            </div>
            
            <h2>📈 Интерактивные графики</h2>
            <div class="files">
                <div class="file-card">
                    <h3>🎮 games_interactive.html</h3>
                    <a href="games_interactive.html" target="_blank">Открыть график</a>
                    <p>Интерактивный график средних оценок по платформам</p>
                    <p class="{ 'success' if os.path.exists(os.path.join(VISUALIZATION_FOLDER, 'games_interactive.html')) else 'warning' }">
                        {'✅ Файл создан' if os.path.exists(os.path.join(VISUALIZATION_FOLDER, 'games_interactive.html')) else '⚠️ Файл не создан'}
                    </p>
                </div>
                
                <div class="file-card">
                    <h3>📊 games_count.html</h3>
                    <a href="games_count.html" target="_blank">Открыть график</a>
                    <p>График количества игр по платформам</p>
                    <p class="{ 'success' if os.path.exists(os.path.join(VISUALIZATION_FOLDER, 'games_count.html')) else 'warning' }">
                        {'✅ Файл создан' if os.path.exists(os.path.join(VISUALIZATION_FOLDER, 'games_count.html')) else '⚠️ Файл не создан'}
                    </p>
                </div>
                
                <div class="file-card">
                    <h3>✨ games_scatter.html</h3>
                    <a href="games_scatter.html" target="_blank">Открыть график</a>
                    <p>Scatter plot распределения оценок</p>
                    <p class="{ 'success' if os.path.exists(os.path.join(VISUALIZATION_FOLDER, 'games_scatter.html')) else 'warning' }">
                        {'✅ Файл создан' if os.path.exists(os.path.join(VISUALIZATION_FOLDER, 'games_scatter.html')) else '⚠️ Файл не создан'}
                    </p>
                </div>
            </div>
            
            <h2>🖼️ Статические изображения</h2>
            <div class="images">
                <div class="image-container">
                    <h3>Основной анализ</h3>
                    <a href="games_analysis.png" target="_blank">
                        <img src="games_analysis.png" alt="Основные графики" onerror="this.style.display='none'">
                    </a>
                    <p><a href="games_analysis.png" target="_blank">games_analysis.png</a></p>
                </div>
                <div class="image-container">
                    <h3>Расширенный анализ</h3>
                    <a href="advanced_analysis.png" target="_blank">
                        <img src="advanced_analysis.png" alt="Расширенный анализ" onerror="this.style.display='none'">
                    </a>
                    <p><a href="advanced_analysis.png" target="_blank">advanced_analysis.png</a></p>
                </div>
            </div>
            
            <div style="margin-top: 40px; padding: 20px; background: #f0f7ff; border-radius: 5px; border-left: 4px solid #2196F3;">
                <h3>ℹ️ Как использовать</h3>
                <p>✅ <strong>Все файлы находятся в одной папке:</strong> {VISUALIZATION_FOLDER}/</p>
                <p>1. Нажмите на любую ссылку выше для открытия файла</p>
                <p>2. HTML файлы открываются в браузере</p>
                <p>3. Для интерактивных графиков Bokeh: наведите курсор, используйте колесико мыши для масштабирования</p>
                <p>4. Если графики не отображаются, установите Bokeh: <code>pip install bokeh</code></p>
            </div>
            
            <div style="margin-top: 30px; text-align: center; color: #666; font-size: 0.9em;">
                <p>Отчет сгенерирован автоматически с помощью Python</p>
                <p>Дата создания: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        </div>
        
        <script>
            // Проверка доступности файлов
            function checkFiles() {{
                const files = ['games_interactive.html', 'games_count.html', 'games_scatter.html'];
                files.forEach(file => {{
                    fetch(file)
                        .then(response => {{
                            if (!response.ok) {{
                                console.log(`Файл ${{file}} не найден`);
                            }}
                        }})
                        .catch(error => {{
                            console.log(`Ошибка при проверке файла ${{file}}:`, error);
                        }});
                }});
            }}
            
            // Запускаем проверку при загрузке страницы
            window.addEventListener('load', checkFiles);
        </script>
    </body>
    </html>
    '''
    
    with open(html_report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML отчет сохранен как '{html_report_path}'")
    
except Exception as e:
    print(f"✗ Ошибка при создании HTML отчета: {e}")

# 6. ФИНАЛЬНЫЙ ОТЧЕТ
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ОТЧЕТ")
print("=" * 60)

# Проверяем какие файлы создались
print("\n📁 ПРОВЕРКА СОЗДАННЫХ ФАЙЛОВ:")
print("-" * 60)

expected_files = [
    'games_interactive.html',
    'games_count.html', 
    'games_scatter.html',
    'games_analysis.png',
    'advanced_analysis.png',
    'processed_games.csv',
    'report.html'
]

all_created = True
for expected_file in expected_files:
    file_path = os.path.join(VISUALIZATION_FOLDER, expected_file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {expected_file}: {size:,} байт")
    else:
        print(f"❌ {expected_file}: ОТСУТСТВУЕТ")
        all_created = False

print("\n" + "=" * 60)
print("АВТОМАТИЧЕСКОЕ ОТКРЫТИЕ ФАЙЛОВ")
print("=" * 60)

# Открываем только те файлы, которые существуют
try:
    # Открываем HTML отчет если он есть
    report_path = os.path.join(VISUALIZATION_FOLDER, 'report.html')
    if os.path.exists(report_path):
        print(f"📄 Открываю HTML отчет...")
        webbrowser.open(f'file://{os.path.abspath(report_path)}')
    else:
        print(f"⚠ HTML отчет не найден")
    
    # Открываем основные HTML файлы если они есть
    html_files_to_check = [
        ('games_interactive.html', '🎮 Интерактивный график'),
        ('games_count.html', '📊 График количества игр'),
        ('games_scatter.html', '✨ Scatter plot')
    ]
    
    for html_file, description in html_files_to_check:
        file_path = os.path.join(VISUALIZATION_FOLDER, html_file)
        if os.path.exists(file_path):
            print(f"{description}...")
            webbrowser.open(f'file://{os.path.abspath(file_path)}')
        else:
            print(f"⚠ {html_file} не найден")
    
    # Открываем PNG файлы если они есть
    png_files = glob.glob(os.path.join(VISUALIZATION_FOLDER, '*.png'))
    for png_file in png_files[:2]:  # Максимум 2 файла
        print(f"🖼️ Открываю изображение: {os.path.basename(png_file)}")
        webbrowser.open(f'file://{os.path.abspath(png_file)}')
    
    print(f"\n📁 Папка с результатами:")
    print(f"   {os.path.abspath(VISUALIZATION_FOLDER)}")
    
    if all_created:
        print("\n✅ ВСЕ ФАЙЛЫ УСПЕШНО СОЗДАНЫ!")
    else:
        print("\n⚠ НЕКОТОРЫЕ ФАЙЛЫ НЕ СОЗДАНЫ:")
        print("   Если отсутствуют HTML файлы Bokeh, выполните:")
        print("   pip install bokeh")
        print("   И запустите скрипт снова")
    
except Exception as e:
    print(f"⚠ Ошибка при открытии файлов: {e}")
    print(f"\n📁 Откройте папку вручную:")
    print(f"   {os.path.abspath(VISUALIZATION_FOLDER)}")

print("=" * 60)