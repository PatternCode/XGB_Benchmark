DATASETS = {
    "adult_income": {
        "source": "openml",
        "openml_name": "adult",
        "target": "class",
        "task": "binary",
    },
    "breast_cancer_wisconsin": {
        "source": "sklearn",
        "loader": "load_breast_cancer",
        "target": "target",
        "task": "binary",
    },
    "dry_bean": {
        "source": "uci",
        "uci_id": 602,
        "target": "Class",
        "task": "multiclass",
    },
    "bank_marketing": {
        "source": "uci",
        "uci_id": 222,
        "target": "y",
        "task": "binary",
    },
    "steel_plates_faults": {
        "source": "uci",
        "uci_id": 198,
        "target": None,
        "task": "multiclass",
    },
    "covertype": {
        "source": "sklearn",
        "loader": "fetch_covtype",
        "target": "target",
        "task": "multiclass",
    },
}
