for me to test and learn concepts, so alot of stuff are scuffed, implementation of a prototype for an improved version of current data pipeline/architecture [here](https://github.com/soonyz06/Barra)

# Data Analysis
* [EDA](src/feature)
  - Basic plots: Missing heatmap, histograms, qq, corr, boxplot
  - More plots: PCA, importances, shap, network graphs, etc
  - OLS plots: Predicted vs Actual, Resid vs Fitted, qq plot, cook's distance, vif, etc
  - Time series: ADF, KPSS, ACF, PACF, etc
* [Previous implementation](https://github.com/soonyz06/Broomburg_Prototype) 
  
 # Broomburg
 ![commands](config/img/help.png)
 * Uses dash for web application (vibe coded cuz i aint learning webdev)
 * Registry pattern to parse and execute commands
 * Able to auto populate [spreadsheets](data/output/fa_models) for DCF
 ![fa](config/img/fa.png)
 ![p](config/img/p.png)
 ![n](config/img/n.png)
 ![txt](config/img/txt.png)
- Potential for semantic chunking and storing in vector database for RAG and stuff

# Risk
  - [Vol](src/utils/risk.py): Close to close, yang zhang and garch  







