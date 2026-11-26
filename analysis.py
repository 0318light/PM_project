from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)


def load_hourly_data(year):
    pm = pd.read_csv(
        DATA_DIR / f"kma_pm10_hourly_{year}.csv",
        encoding="cp949",
        usecols=[0, 1, 2, 3],
        low_memory=False,
    )
    pm.columns = ["station_code", "station_name", "datetime", "pm10"]

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

    pm["datetime"] = pd.to_datetime(pm["datetime"], errors="coerce")
    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    pm["pm10"] = pd.to_numeric(pm["pm10"], errors="coerce")

    for column in ["precipitation", "wind_speed", "wind_direction"]:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    merged = pm.merge(
        weather[
            ["station_code", "datetime", "precipitation", "wind_speed", "wind_direction"]
        ],
        on=["station_code", "datetime"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.dropna(subset=["datetime", "pm10"]).sort_values(
        ["station_code", "datetime"]
    )
    merged["precipitation"] = merged["precipitation"].fillna(0)
    for column in ["wind_speed", "wind_direction"]:
        merged[column] = merged[column].fillna(
            merged.groupby("station_code")[column].transform("median")
        )
        merged[column] = merged[column].fillna(merged[column].median())

    merged["lag1_pm10"] = merged.groupby("station_code")["pm10"].shift(1)
    merged["lag24_pm10"] = merged.groupby("station_code")["pm10"].shift(24)
    merged["rolling24_pm10"] = merged.groupby("station_code")[
        "pm10"
    ].transform(lambda series: series.shift(1).rolling(24).mean())
    merged["hour"] = merged["datetime"].dt.hour
    merged["month"] = merged["datetime"].dt.month
    merged["dayofweek"] = merged["datetime"].dt.dayofweek
    merged["target_next_pm10"] = merged.groupby("station_code")["pm10"].shift(-1)

    return merged.dropna(
        subset=[
            "lag1_pm10",
            "lag24_pm10",
            "rolling24_pm10",
            "target_next_pm10",
        ]
    )


def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def make_notebook():
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
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
            "\n"
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
        nbf.v4.new_markdown_cell("## 2. 모델링\n\n2023년으로 학습하고 2024년으로 평가해 연도 간 일반화 성능을 확인합니다."),
        nbf.v4.new_code_cell(
            "features = ['station_code','lag1_pm10','lag24_pm10','rolling24_pm10','precipitation','wind_speed','wind_direction','hour','month','dayofweek']\n"
            "X_train = train[features]\n"
            "y_train = train['target_next_pm10']\n"
            "X_test = test[features]\n"
            "y_test = test['target_next_pm10']\n"
            "\n"
            "models = {\n"
            "    'Linear Regression': LinearRegression(),\n"
            "    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42),\n"
            "    'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42),\n"
            "}\n"
            "results = {}\n"
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
        nbf.v4.new_markdown_cell("## 3. 성능 시각화"),
        nbf.v4.new_code_cell(
            "result_df = pd.DataFrame(results).T.sort_values('RMSE')\n"
            "ax = result_df['RMSE'].plot(kind='bar', title='Model RMSE Comparison')\n"
            "ax.set_ylabel('RMSE')\n"
            "plt.tight_layout()\n"
            "plt.savefig(IMAGE_DIR / 'model_comparison.png', dpi=150)\n"
            "plt.show()\n"
            "result_df"
        ),
        nbf.v4.new_markdown_cell("## 4. 특성 중요도"),
        nbf.v4.new_code_cell(
            "tree_model = models['Random Forest']\n"
            "importance = pd.Series(tree_model.feature_importances_, index=features).sort_values(ascending=False)\n"
            "ax = importance.head(10).plot(kind='barh', title='Random Forest Feature Importance')\n"
            "ax.invert_yaxis()\n"
            "plt.tight_layout()\n"
            "plt.savefig(IMAGE_DIR / 'feature_importance.png', dpi=150)\n"
            "plt.show()\n"
            "importance"
        ),
    ]
    notebook["cells"] = cells
    nbf.write(notebook, ROOT / "final_project.ipynb")


def main():
    make_notebook()
    train = load_hourly_data(2023)
    test = load_hourly_data(2024)

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

    results = {}
    predictions = test[["station_code", "station_name", "datetime"]].copy()
    predictions["actual_next_pm10"] = y_test

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results[name] = calculate_metrics(y_test, prediction)
        predictions[name] = prediction

    result_df = pd.DataFrame(results).T
    result_df.to_csv(ROOT / "model_results.csv", encoding="utf-8-sig")
    predictions.to_csv(ROOT / "predictions_2024.csv", index=False, encoding="utf-8-sig")
    train.to_csv(ROOT / "data" / "model_ready_2023.csv", index=False, encoding="utf-8-sig")
    test.to_csv(ROOT / "data" / "model_ready_2024.csv", index=False, encoding="utf-8-sig")

    ax = result_df["RMSE"].sort_values().plot(
        kind="bar", title="Model RMSE Comparison"
    )
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison.png", dpi=150)
    plt.close()

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

    print("train rows:", len(train))
    print("test rows:", len(test))
    print(result_df.to_string())
    print("overlap station count:", len(set(train.station_code) & set(test.station_code)))


if __name__ == "__main__":
    main()
