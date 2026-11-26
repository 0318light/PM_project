from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 프로젝트 폴더를 기준으로 데이터와 그래프 저장 위치를 정합니다.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)


def load_hourly_data(year):
    """PM10·기상 자료를 결합하고 모델 입력 변수와 예측 대상을 만듭니다."""
    # 해당 연도의 PM10 관측 자료를 읽고 컬럼명을 통일합니다.
    pm = pd.read_csv(
        DATA_DIR / f"kma_pm10_hourly_{year}.csv",
        encoding="cp949",
        usecols=[0, 1, 2, 3],
        low_memory=False,
    )
    pm.columns = ["station_code", "station_name", "datetime", "pm10"]

    # 같은 연도의 ASOS 자료에서 분석에 사용할 기상 변수만 불러옵니다.
    weather = pd.read_csv(
        DATA_DIR / f"kma_asos_hourly_{year}.csv",
        encoding="cp949",
        usecols=[0, 1, 2, 3, 4, 5],
        low_memory=False,
    )
    weather.columns = [
        "station_code",
        "station_name",
        "datetime",
        "precipitation",
        "wind_speed",
        "wind_direction",
    ]

    # 날짜와 측정값을 계산 가능한 형식으로 변환합니다.
    pm["datetime"] = pd.to_datetime(pm["datetime"], errors="coerce")
    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    pm["pm10"] = pd.to_numeric(pm["pm10"], errors="coerce")

    # 기상 측정값도 숫자로 변환하고, 변환할 수 없는 값은 결측치로 둡니다.
    for column in ["precipitation", "wind_speed", "wind_direction"]:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    # 관측소 코드와 관측 시각이 모두 같은 PM10·기상 자료를 결합합니다.
    # validate는 기상 자료에 같은 관측소·시각 조합이 중복되는 경우 오류를 알려줍니다.
    merged = pm.merge(
        weather[
            ["station_code", "datetime", "precipitation", "wind_speed", "wind_direction"]
        ],
        on=["station_code", "datetime"],
        how="inner",
        validate="many_to_one",
    )

    # 날짜나 PM10이 없는 행을 제외하고 관측소별 시간순으로 정렬합니다.
    merged = merged.dropna(subset=["datetime", "pm10"]).sort_values(
        ["station_code", "datetime"]
    )

    # 강수량 결측은 0으로 채우고, 풍속·풍향은 관측소별 중앙값으로 대체합니다.
    merged["precipitation"] = merged["precipitation"].fillna(0)
    for column in ["wind_speed", "wind_direction"]:
        merged[column] = merged[column].fillna(
            merged.groupby("station_code")[column].transform("median")
        )
        merged[column] = merged[column].fillna(merged[column].median())

    # 관측소별 이전 PM10 값을 입력 변수로 사용합니다.
    merged["lag1_pm10"] = merged.groupby("station_code")["pm10"].shift(1)
    merged["lag24_pm10"] = merged.groupby("station_code")["pm10"].shift(24)

    # 현재 시각 정보가 섞이지 않도록 직전 24시간의 평균을 계산합니다.
    merged["rolling24_pm10"] = merged.groupby("station_code")[
        "pm10"
    ].transform(lambda series: series.shift(1).rolling(24).mean())

    # 일중·월별·요일별 패턴을 모델이 학습할 수 있도록 시간 정보를 나눕니다.
    merged["hour"] = merged["datetime"].dt.hour
    merged["month"] = merged["datetime"].dt.month
    merged["dayofweek"] = merged["datetime"].dt.dayofweek

    # 예측 목표는 같은 관측소에서 관측된 다음 시간 PM10입니다.
    merged["target_next_pm10"] = merged.groupby("station_code")["pm10"].shift(-1)

    # 과거 입력값이나 다음 시간 정답이 없는 행은 모델 학습에서 제외합니다.
    return merged.dropna(
        subset=[
            "lag1_pm10",
            "lag24_pm10",
            "rolling24_pm10",
            "target_next_pm10",
        ]
    )


def calculate_metrics(y_true, y_pred):
    """예측값의 RMSE, MAE, R²를 계산해 결과 표에 사용할 형태로 반환합니다."""
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def create_eda_visualizations(data):
    """학습 자료의 수치형 변수 분포와 상관계수 히트맵을 저장합니다."""
    columns = [
        "target_next_pm10",
        "lag1_pm10",
        "lag24_pm10",
        "rolling24_pm10",
        "precipitation",
        "wind_speed",
    ]
    correlation = data[columns].corr()
    labels = [
        "Next PM10",
        "Lag 1 PM10",
        "Lag 24 PM10",
        "Rolling 24 PM10",
        "Precipitation",
        "Wind speed",
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    ax.set_title("2023 Training Data: Pearson Correlation")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    distributions = [
        ("target_next_pm10", "Next-hour PM10 (ug/m3)", False),
        ("lag1_pm10", "Previous-hour PM10 (ug/m3)", False),
        ("precipitation", "log(1 + precipitation [mm])", True),
        ("wind_speed", "Wind speed (m/s)", False),
    ]
    for ax, (column, label, log_transform) in zip(axes.flat, distributions):
        values = np.log1p(data[column]) if log_transform else data[column]
        ax.hist(values, bins=40, color="#3B82A0", edgecolor="white")
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel(label)
        ax.set_ylabel("Observations")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("2023 Training Data: Variable Distributions")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "eda_distribution.png", dpi=150)
    plt.close(fig)


def create_actual_vs_predicted_plot(actual, predicted):
    """실제값과 Extra Trees 예측값을 비교하는 회귀 산점도를 저장합니다."""
    actual = pd.Series(actual)
    predicted = pd.Series(predicted, index=actual.index)
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        actual,
        predicted,
        s=8,
        alpha=0.15,
        color="#3B82A0",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="#C43C39",
        label="Ideal prediction",
    )
    ax.set(
        title="Actual vs Predicted PM10 (Extra Trees, 2024)",
        xlabel="Actual PM10 (ug/m3)",
        ylabel="Predicted PM10 (ug/m3)",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def make_notebook():
    """분석 절차를 설명하는 Jupyter 노트북 파일을 생성합니다."""
    # 노트북에서 사용할 Python 커널 정보를 설정합니다.
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }

    cells = [
        nbf.v4.new_markdown_cell(
            "# 기상·지역 요인을 고려한 국내 미세먼지 농도 분석과 예측\n\n"
            "KMA ASOS 기상자료와 PM10 부유분진 자료를 관측소 코드와 시각으로 결합한 뒤,\n"
            "다음 시간 PM10을 예측합니다."
        ),
        nbf.v4.new_code_cell(
            "# 데이터 처리, 시각화, 회귀 모델 학습에 필요한 라이브러리를 불러옵니다.\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
            "\n"
            "# 노트북이 실행되는 폴더를 기준으로 데이터와 그래프 경로를 지정합니다.\n"
            "ROOT = Path('.')\n"
            "DATA_DIR = ROOT / 'data'\n"
            "IMAGE_DIR = ROOT / 'images'\n"
            "IMAGE_DIR.mkdir(exist_ok=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. 데이터 결합\n\n"
            "2023년과 2024년 자료를 같은 관측소 코드와 시각 기준으로 결합합니다."
        ),
        nbf.v4.new_code_cell(
            "# 연도별 미세먼지와 기상 자료를 불러와 하나로 정리하는 함수입니다.\n"
            "def load_year(year):\n"
            "    # 해당 연도의 PM10 관측 파일에서 분석에 필요한 열만 읽습니다.\n"
            "    pm = pd.read_csv(DATA_DIR / f'kma_pm10_hourly_{year}.csv', encoding='cp949', usecols=[0, 1, 2, 3])\n"
            "    pm.columns = ['station_code', 'station_name', 'datetime', 'pm10']\n"
            "\n"
            "    # 같은 연도의 ASOS 자료에서 분석에 필요한 기상 열만 읽습니다.\n"
            "    wx = pd.read_csv(DATA_DIR / f'kma_asos_hourly_{year}.csv', encoding='cp949', usecols=[0, 1, 2, 3, 4, 5])\n"
            "    wx.columns = ['station_code', 'station_name', 'datetime', 'precipitation', 'wind_speed', 'wind_direction']\n"
            "\n"
            "    # 날짜와 관측값을 계산에 사용할 수 있는 형식으로 변환합니다.\n"
            "    pm['datetime'] = pd.to_datetime(pm['datetime'], errors='coerce')\n"
            "    wx['datetime'] = pd.to_datetime(wx['datetime'], errors='coerce')\n"
            "    pm['pm10'] = pd.to_numeric(pm['pm10'], errors='coerce')\n"
            "\n"
            "    # 기상 관측값도 숫자로 변환합니다. 변환할 수 없는 값은 결측치가 됩니다.\n"
            "    for c in ['precipitation', 'wind_speed', 'wind_direction']:\n"
            "        wx[c] = pd.to_numeric(wx[c], errors='coerce')\n"
            "\n"
            "    # 관측소 코드와 관측 시각이 모두 같은 자료끼리 결합합니다.\n"
            "    weather_columns = ['station_code', 'datetime', 'precipitation', 'wind_speed', 'wind_direction']\n"
            "    df = pm.merge(wx[weather_columns], on=['station_code', 'datetime'], how='inner')\n"
            "\n"
            "    # 날짜 또는 PM10이 없는 행을 제외하고 관측소별 시간순으로 정렬합니다.\n"
            "    df = df.dropna(subset=['datetime', 'pm10'])\n"
            "    df = df.sort_values(['station_code', 'datetime'])\n"
            "\n"
            "    # 강수량 결측은 비가 오지 않은 것으로 보고 0으로 채웁니다.\n"
            "    df['precipitation'] = df['precipitation'].fillna(0)\n"
            "\n"
            "    # 풍속·풍향은 먼저 관측소별 중앙값으로 채우고,\n"
            "    # 해당 관측소 값도 없으면 전체 자료의 중앙값을 사용합니다.\n"
            "    for c in ['wind_speed','wind_direction']:\n"
            "        station_median = df.groupby('station_code')[c].transform('median')\n"
            "        df[c] = df[c].fillna(station_median)\n"
            "        df[c] = df[c].fillna(df[c].median())\n"
            "\n"
            "    # 직전 1시간 및 24시간의 PM10 농도를 과거 정보 변수로 만듭니다.\n"
            "    df['lag1_pm10'] = df.groupby('station_code')['pm10'].shift(1)\n"
            "    df['lag24_pm10'] = df.groupby('station_code')['pm10'].shift(24)\n"
            "\n"
            "    # 현재 시각 정보가 섞이지 않도록 직전 24시간 평균을 계산합니다.\n"
            "    df['rolling24_pm10'] = df.groupby('station_code')['pm10'].transform(\n"
            "        lambda values: values.shift(1).rolling(24).mean()\n"
            "    )\n"
            "\n"
            "    # 시간·월·요일을 분리해 시간대 및 계절 패턴을 모델에 제공합니다.\n"
            "    df['hour'] = df['datetime'].dt.hour\n"
            "    df['month'] = df['datetime'].dt.month\n"
            "    df['dayofweek'] = df['datetime'].dt.dayofweek\n"
            "\n"
            "    # 정답값은 현재가 아니라 같은 관측소의 다음 시간 PM10으로 지정합니다.\n"
            "    df['target_next_pm10'] = df.groupby('station_code')['pm10'].shift(-1)\n"
            "\n"
            "    # 과거 입력값이나 다음 시간 정답이 없는 행은 학습에서 제외합니다.\n"
            "    required_columns = ['lag1_pm10', 'lag24_pm10', 'rolling24_pm10', 'target_next_pm10']\n"
            "    return df.dropna(subset=required_columns)\n"
            "\n"
            "# 2023년은 학습용, 2024년은 연도 간 성능 평가용 자료로 불러옵니다.\n"
            "train = load_year(2023)\n"
            "test = load_year(2024)\n"
            "\n"
            "# 전처리 결과를 행·열 개수로 확인합니다.\n"
            "print('train:', train.shape, 'test:', test.shape)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. 탐색적 데이터 분석 (EDA)\n\n"
            "모델 학습에 사용하는 2023년 자료의 주요 변수 분포와 Pearson 상관관계를 확인합니다. "
            "관측소 코드·시간 변수는 식별자 또는 주기형 변수이므로 상관계수에서 제외하고, "
            "풍향은 원형 변수라 선형 상관계수에 포함하지 않습니다."
        ),
        nbf.v4.new_code_cell(
            "# 타깃과 연속형 입력 변수의 기술통계량을 확인합니다.\n"
            "eda_columns = ['target_next_pm10', 'lag1_pm10', 'lag24_pm10', 'rolling24_pm10', 'precipitation', 'wind_speed']\n"
            "train[eda_columns].describe().T[['mean', 'std', 'min', '50%', 'max']].round(3)\n"
            "\n"
            "# 분석 결과를 README에서 사용할 이미지 파일로 저장합니다.\n"
            "import numpy as np\n"
            "correlation = train[eda_columns].corr()\n"
            "labels = ['Next PM10', 'Lag 1 PM10', 'Lag 24 PM10', 'Rolling 24 PM10', 'Precipitation', 'Wind speed']\n"
            "fig, ax = plt.subplots(figsize=(9, 7))\n"
            "image = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)\n"
            "ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha='right')\n"
            "ax.set_yticks(range(len(labels)), labels=labels)\n"
            "for row in range(len(labels)):\n"
            "    for column in range(len(labels)):\n"
            "        ax.text(column, row, f'{correlation.iloc[row, column]:.2f}', ha='center', va='center', fontsize=9)\n"
            "ax.set_title('2023 Training Data: Pearson Correlation')\n"
            "fig.colorbar(image, ax=ax, label='Pearson correlation')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'correlation_heatmap.png', dpi=150)\n"
            "plt.show()\n"
            "\n"
            "# 편중된 강수량 분포는 log(1+x) 변환해 나머지 분포와 비교합니다.\n"
            "fig, axes = plt.subplots(2, 2, figsize=(11, 8))\n"
            "distributions = [('target_next_pm10', 'Next-hour PM10 (ug/m3)', False), ('lag1_pm10', 'Previous-hour PM10 (ug/m3)', False), ('precipitation', 'log(1 + precipitation [mm])', True), ('wind_speed', 'Wind speed (m/s)', False)]\n"
            "for ax, (column, label, log_transform) in zip(axes.flat, distributions):\n"
            "    values = np.log1p(train[column]) if log_transform else train[column]\n"
            "    ax.hist(values, bins=40, color='#3B82A0', edgecolor='white')\n"
            "    ax.set_title(column.replace('_', ' ').title())\n"
            "    ax.set_xlabel(label)\n"
            "    ax.set_ylabel('Observations')\n"
            "    ax.grid(axis='y', alpha=0.25)\n"
            "fig.suptitle('2023 Training Data: Variable Distributions')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'eda_distribution.png', dpi=150)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 3. 모델링\n\n2023년으로 학습하고 2024년으로 평가해 연도 간 일반화 성능을 확인합니다."),
        nbf.v4.new_code_cell(
            "# 모델 학습에 사용할 입력 변수와 예측할 정답 변수를 지정합니다.\n"
            "features = ['station_code','lag1_pm10','lag24_pm10','rolling24_pm10','precipitation','wind_speed','wind_direction','hour','month','dayofweek']\n"
            "X_train = train[features]\n"
            "y_train = train['target_next_pm10']\n"
            "X_test = test[features]\n"
            "y_test = test['target_next_pm10']\n"
            "\n"
            "# 선형 모델과 트리 기반 모델의 예측 성능을 비교합니다.\n"
            "models = {\n"
            "    'Linear Regression': LinearRegression(),\n"
            "    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42),\n"
            "    'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42),\n"
            "}\n"
            "results = {}\n"
            "\n"
            "# 각 모델을 학습하고 2024년 자료의 예측값과 평가 지표를 저장합니다.\n"
            "for name, model in models.items():\n"
            "    model.fit(X_train, y_train)\n"
            "    pred = model.predict(X_test)\n"
            "    results[name] = {\n"
            "        'RMSE': mean_squared_error(y_test, pred) ** 0.5,\n"
            "        'MAE': mean_absolute_error(y_test, pred),\n"
            "        'R2': r2_score(y_test, pred),\n"
            "    }\n"
            "pd.DataFrame(results).T"
        ),
        nbf.v4.new_markdown_cell("## 4. 성능 시각화"),
        nbf.v4.new_code_cell(
            "# RMSE가 낮은 모델부터 정렬해 성능을 막대그래프로 비교합니다.\n"
            "result_df = pd.DataFrame(results).T.sort_values('RMSE')\n"
            "ax = result_df['RMSE'].plot(kind='bar', title='Model RMSE Comparison')\n"
            "ax.set_ylabel('RMSE')\n"
            "plt.tight_layout()\n"
            "# 그래프를 README에서 사용할 수 있도록 이미지 파일로 저장합니다.\n"
            "plt.savefig(IMAGE_DIR / 'model_comparison.png', dpi=150)\n"
            "plt.show()\n"
            "result_df"
        ),
        nbf.v4.new_code_cell(
            "# 최종 선택한 Extra Trees 모델로 테스트 데이터의 PM10 예측값을 계산합니다.\n"
            "extra_trees_pred = models['Extra Trees'].predict(X_test)\n"
            "# 실제값과 예측값 중 작은 값을 두 축의 공통 최솟값으로 정합니다.\n"
            "lower = min(y_test.min(), extra_trees_pred.min())\n"
            "# 실제값과 예측값 중 큰 값을 두 축의 공통 최댓값으로 정합니다.\n"
            "upper = max(y_test.max(), extra_trees_pred.max())\n"
            "# 8x8인 그림과 그래프 축 객체를 만듭니다.\n"
            "fig, ax = plt.subplots(figsize=(8, 8))\n"
            "# 실제값을 x축, 예측값을 y축에 그리고 점을 작고 투명하게 표시합니다.\n"
            "ax.scatter(y_test, extra_trees_pred, s=8, alpha=0.15, color='#3B82A0', edgecolors='none', rasterized=True)\n"
            "# 예측이 실제값과 같은 y=x 기준선을 빨간 점선으로 표시합니다.\n"
            "ax.plot([lower, upper], [lower, upper], linestyle='--', color='#C43C39', label='Ideal prediction')\n"
            "# 그래프 제목, 축 이름, 두 축의 동일한 표시 범위를 지정합니다.\n"
            "ax.set(title='Actual vs Predicted PM10 (Extra Trees, 2024)', xlabel='Actual PM10 (ug/m3)', ylabel='Predicted PM10 (ug/m3)', xlim=(lower, upper), ylim=(lower, upper))\n"
            "# x축과 y축의 단위 길이를 같게 해 기준선 비교가 왜곡되지 않게 합니다.\n"
            "ax.set_aspect('equal', adjustable='box')\n"
            "# 점의 위치를 읽기 쉽도록 격자선을 표시합니다.\n"
            "ax.grid(alpha=0.25)\n"
            "# 기준선 설명을 그래프 범례에 표시합니다.\n"
            "ax.legend()\n"
            "# 제목이나 축 이름이 잘리지 않도록 여백을 자동 조정합니다.\n"
            "fig.tight_layout()\n"
            "# 그래프를 README에서 연결할 PNG 파일로 저장합니다.\n"
            "fig.savefig(IMAGE_DIR / 'actual_vs_predicted.png', dpi=150)\n"
            "# 노트북 출력에 그래프를 표시합니다.\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 5. 특성 중요도"),
        nbf.v4.new_code_cell(
            "# Random Forest가 계산한 특성 중요도를 내림차순으로 정리합니다.\n"
            "tree_model = models['Random Forest']\n"
            "importance = pd.Series(tree_model.feature_importances_, index=features).sort_values(ascending=False)\n"
            "\n"
            "# 중요도가 높은 상위 10개 변수를 가로 막대그래프로 나타냅니다.\n"
            "ax = importance.head(10).plot(kind='barh', title='Random Forest Feature Importance')\n"
            "ax.invert_yaxis()\n"
            "plt.tight_layout()\n"
            "# 특성 중요도 그래프도 이미지 파일로 저장합니다.\n"
            "plt.savefig(IMAGE_DIR / 'feature_importance.png', dpi=150)\n"
            "plt.show()\n"
            "importance"
        ),
    ]
    notebook["cells"] = cells
    nbf.write(notebook, ROOT / "final_project.ipynb")


def main():
    """두 연도 자료를 처리하고 모델 성능과 시각화 결과를 저장합니다."""
    # 실행 가능한 분석 절차도 함께 노트북 파일로 만듭니다.
    make_notebook()

    # 2023년 자료로 학습하고 2024년 자료로 성능을 평가합니다.
    train = load_hourly_data(2023)
    test = load_hourly_data(2024)
    create_eda_visualizations(train)

    eda_columns = [
        "target_next_pm10",
        "lag1_pm10",
        "lag24_pm10",
        "rolling24_pm10",
        "precipitation",
        "wind_speed",
    ]
    print("2023 EDA summary:")
    print(
        train[eda_columns]
        .describe()
        .T[["mean", "std", "min", "50%", "max"]]
        .round(3)
        .to_string()
    )

    # 예측에 사용할 입력 변수 목록과 다음 시간 PM10 정답을 분리합니다.
    features = [
        "station_code",
        "lag1_pm10",
        "lag24_pm10",
        "rolling24_pm10",
        "precipitation",
        "wind_speed",
        "wind_direction",
        "hour",
        "month",
        "dayofweek",
    ]
    X_train = train[features]
    y_train = train["target_next_pm10"]
    X_test = test[features]
    y_test = test["target_next_pm10"]

    # 서로 다른 방식의 회귀 모델을 같은 학습·평가 자료로 비교합니다.
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }

    # 평가 지표와 관측값·모델별 예측값을 모아둘 공간을 준비합니다.
    results = {}
    predictions = test[["station_code", "station_name", "datetime"]].copy()
    predictions["actual_next_pm10"] = y_test

    # 각 모델을 학습하고 2024년 자료를 예측한 뒤 성능을 계산합니다.
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results[name] = calculate_metrics(y_test, prediction)
        predictions[name] = prediction

    # 모델별 성능과 상세 예측 결과를 CSV 파일로 저장합니다.
    result_df = pd.DataFrame(results).T
    result_df.to_csv(ROOT / "model_results.csv", encoding="utf-8-sig")
    predictions.to_csv(ROOT / "predictions_2024.csv", index=False, encoding="utf-8-sig")

    # 전처리된 학습·평가 자료도 재사용할 수 있도록 각각 저장합니다.
    train.to_csv(ROOT / "data" / "model_ready_2023.csv", index=False, encoding="utf-8-sig")
    test.to_csv(ROOT / "data" / "model_ready_2024.csv", index=False, encoding="utf-8-sig")

    # 모델별 RMSE 비교 그래프를 만들고 이미지로 저장합니다.
    ax = result_df["RMSE"].sort_values().plot(
        kind="bar", title="Model RMSE Comparison"
    )
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # Random Forest의 특성 중요도를 계산해 상위 10개를 그래프로 저장합니다.
    importance = pd.Series(
        models["Random Forest"].feature_importances_,
        index=features,
    ).sort_values(ascending=False)
    ax = importance.head(10).sort_values().plot(
        kind="barh", title="Random Forest Feature Importance"
    )
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # 최종 선택 모델인 Extra Trees의 실제값과 예측값을 산점도로 저장합니다.
    create_actual_vs_predicted_plot(
        y_test,
        predictions["Extra Trees"],
    )

    # 분석에 사용된 행 수와 모델별 성능을 화면에 요약합니다.
    print("train rows:", len(train))
    print("test rows:", len(test))
    print(result_df.to_string())
    print("overlap station count:", len(set(train.station_code) & set(test.station_code)))


if __name__ == "__main__":
    main()
