class DataAnalyzer:
    def __init__(self, df, target=None):
        self.df = df
        self.target = target

    def analyze(self):
        df = self.df

        report = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict(),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in df.select_dtypes(include="object").columns
            },
        }

        return report