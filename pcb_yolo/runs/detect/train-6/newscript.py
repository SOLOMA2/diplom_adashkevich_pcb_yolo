from ultralytics.utils.plotting import plot_results
from pathlib import Path

# Укажи путь к своему файлу results.csv
csv_path = Path(r'D:\YOLO-PCB.Финал\YOLO-PCB\runs\detect\train-6\results.csv')
plot_results(file=csv_path) 
print("График results.png успешно создан в той же папке!")
