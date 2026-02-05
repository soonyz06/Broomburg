from fredapi import Fred
import oecddatabuilder as OECD_data

from src.io.fetch import load_apikeys

api_keys = load_apikeys()



sources = {"fred": api_keys.get("FRED_API_KEY", None)}
fred = Fred(api_key=sources["fred"])

df = fred.get_series_all_releases('GDP')

print(df.head())
df.to_csv("fred.csv")
"""

recipe_loader = OECD_data.RecipeLoader()
default_recipe = recipe_loader.load(recipe_name="DEFAULT")
builder = OECD_data.OECDAPI_Databuilder(config=default_recipe, start="1990-Q1", end="2024-Q4", freq="Q", response_format="csv",
                                    dbpath="../datasets/OECD",
                                    base_url="https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1/", request_interval=60)
df = builder.create_dataframe()
print(df.head())
df.to_csv("oecd.csv")
"""
