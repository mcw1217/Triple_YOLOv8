from ultralytics import YOLO
import pandas as pd

def extract_metrics(results, class_names):
    data = {
        'Class': class_names,
        'Precision': [round(p, 3) for p in results.box.p.tolist()],
        'Recall': [round(r, 3) for r in results.box.r.tolist()],
        'F1_Score': [round(f1, 3) for f1 in results.box.f1.tolist()],
        'AP50': [round(ap50, 3) for ap50 in results.box.ap50.tolist()],
        'AP50-95': [round(ap, 3) for ap in results.box.ap.tolist()]
    }
    return data

def highlight_max(df):
    # 각 컬럼의 최대값 찾기
    max_vals = df.max(numeric_only=True)
    
    # 최대값에 'B' 문자열 추가
    for col in df.columns:
        if col in max_vals.index:
            df[col] = df[col].apply(lambda x: f'B{x}' if x == max_vals[col] else x)
    return df

def add_average_row(df):
    # 숫자형 데이터만 추출
    numeric_df = df.applymap(lambda x: float(str(x).replace('B', '')) if isinstance(x, str) else x)
    
    # 각 모델별 평균 계산
    avg_row = numeric_df.mean(axis=0).round(3).to_frame().T
    avg_row.index = ['Average']
    
    # 평균 행을 원본 데이터프레임에 추가
    return pd.concat([df, avg_row])

# 모델 경로와 이름 정의
if __name__ == '__main__':
    dataset_name = "new_data"
    model_paths = {
        # 'YOLOv5n': f'EX_last/{dataset_name}/train/yolov5n/weights/best.pt',
        # 'YOLOv5s': f'EX_last/{dataset_name}/train/yolov5s/weights/best.pt',
        # 'YOLOv5m': f'EX_last/{dataset_name}/train/yolov5m/weights/best.pt',
        # 'YOLOv5l': f'EX_last/{dataset_name}/train/yolov5l/weights/best.pt',
        # 'YOLOv5x': f'EX_last/{dataset_name}/train/yolov5x/weights/best.pt',
        'YOLOv8n': f'EX/{dataset_name}/train/yolov8n34/weights/best.pt',
        # 'YOLOv8s': f'EX_last/{dataset_name}/train/yolov8s/weights/best.pt',
        # 'YOLOv8m': f'EX_last/{dataset_name}/train/yolov8m/weights/best.pt',
        # 'YOLOv8l': f'EX_last/{dataset_name}/train/yolov8l/weights/best.pt',
        # 'YOLOv8x': f'EX_last/{dataset_name}/train/yolov8x/weights/best.pt',
        # 'YOLOv9c': f'EX_last/{dataset_name}/train/yolov9c/weights/best.pt',
        # 'YOLOv9e': f'EX_last/{dataset_name}/train/yolov9e/weights/best.pt',
    }

    # 빈 데이터프레임 초기화
    df_total = pd.DataFrame()

    # 각 모델에 대해 평가 수행 및 데이터프레임 생성
    for model_name, model_path in model_paths.items():
        # 모델 로드 및 평가 수행
        model = YOLO(model_path)
        results = model.val(data=f"yaml/{dataset_name}_test.yaml", visualize=True, batch=1)

        # 클래스 이름 목록 추출
        class_names = list(results.names.values())

        # 필요한 평가 지표 추출
        data = extract_metrics(results, class_names)
        
        # 데이터 프레임 생성
        df = pd.DataFrame(data)
        df['Model'] = model_name
        
        # 각 모델에 대한 결과를 합치기 위해 피벗 테이블 생성
        df_pivot = df.pivot(index='Model', columns='Class')
        df_pivot.columns = [f'{metric}_{cls}' for metric, cls in df_pivot.columns]
        
        # 총 데이터프레임에 추가
        df_total = pd.concat([df_total, df_pivot], axis=0)

    # 각 컬럼의 최대값을 강조
    df_total = highlight_max(df_total)

    # 각 모델별 평균 값을 추가
    df_total_with_avg = add_average_row(df_total)

    # 결과를 엑셀 파일로 저장
    output_file = 'YOLO_models_evaluation.xlsx'
    df_total_with_avg.to_excel(output_file, index=True)
    print(f"Results have been saved to {output_file}")

    # 각 모델별 클래스별 F1 Score 및 평균 출력
    for model in df_total_with_avg.index.unique():
        if model != 'Average':
            print(f"F1 Scores for {model}:")
            model_df = df_total_with_avg.loc[model]
            f1_scores = [float(model_df[col].replace('B', '')) if isinstance(model_df[col], str) else model_df[col] for col in model_df.index if col.startswith('F1_Score')]
            print(f1_scores)
            print(f"Mean F1 Score for {model}: {sum(f1_scores) / len(f1_scores)}\n")
