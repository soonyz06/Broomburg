APP = { #use by others
    "nlp": {
        "llm": ["source", "reasoning", "query decomposition", "entity decomposition", "loop"], #distill model, teach me qt
        "rag": ["upload", "query+keyword search", "web search", "cache search", "dynamic search", "store consolidated notes"], #context learning
        "actuators": ["call functions", "sentiment"],
        "orchestration": ["sequential", "hierarchical", "parallel"]
    },
    "ml": {
        "data": ["eda", "cleaning", "preprocessing", "feature construction", "feature selection", "modelling"]
    },
    "productivity": { #llm to emulate workflow (treat as human)
        "notes": ["targ+subtags", "adjacency list with set", "free-moving network graph with forces (center, repel and link)"],
        "todo": ["priority", "calender", "recurring", "completed"],
        "clock": ["timer", "alarm", "logger", "timespent"],
        "storage": ["upload", "download", "query"],
        "inbox": [],
        "bookmark": []
    },
    "financial": {
        "cached": ["display", "export", "import"],
        "live": ["wti", "watchlist", "heatmap", "chart"]
    }
}

RAW = {
    "macro": ["fred", "oecd", "wb"],
    "news": ["rss", "guardian", "8k", "s-1"],
    "analyst": ["ratings", "earnings", "estimates"],
    "alt": ["google trends", "3/4/5", "13F/D/G"],
    "high-level": ["databento"]
}     

#read: input latest_by_identifiers and output partition_cols
#save: input df requires partition_cols in df.columns and source, it adds partition_cols to filedir

    
