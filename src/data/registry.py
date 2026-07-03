DATASETS = {
    "adult_income": {
        "name": "Adult Income",
        "domain": "census",
        "task": "binary",
        "target": "class",
        "size": "small",
        "source": {
            "type": "openml",
            "name": "adult",
            "version": 2,
        },
    },

    "breast_cancer_wisconsin": {
        "name": "Breast Cancer Wisconsin",
        "domain": "healthcare",
        "task": "binary",
        "target": "target",
        "size": "small",
        "source": {
            "type": "sklearn",
            "loader": "load_breast_cancer",
        },
    },

    "dry_bean": {
        "name": "Dry Bean",
        "domain": "agriculture",
        "task": "multiclass",
        "target": "Class",
        "size": "small",
        "source": {
            "type": "uci",
            "id": 602,
        },
    },

    "bank_marketing": {
        "name": "Bank Marketing",
        "domain": "finance",
        "task": "binary",
        "target": "y",
        "size": "medium",
        "source": {
            "type": "uci",
            "id": 222,
        },
    },

    "steel_plates_faults": {
        "name": "Steel Plates Faults",
        "domain": "manufacturing",
        "task": "multiclass",
        "target": "fault_type",
        "size": "small",
        "source": {
            "type": "uci",
            "id": 198,
        },
        "notes": "Original target is represented by multiple one-hot fault columns.",
    },

    "covertype": {
        "name": "Covertype",
        "domain": "forestry",
        "task": "multiclass",
        "target": "target",
        "size": "large",
        "source": {
            "type": "sklearn",
            "loader": "fetch_covtype",
        },
    },

    "credit_card_fraud": {
        "name": "Credit Card Fraud",
        "domain": "finance",
        "task": "binary",
        "target": "Class",
        "size": "large",
        "source": {
            "type": "kaggle",
            "slug": "mlg-ulb/creditcardfraud",
            "file_path": "creditcard.csv",
            "requires_auth": True,
        },
    },



    "unsw_nb15": {
        "name": "UNSW-NB15",
        "domain": "cybersecurity",
        "task": "multiclass",
        "target": "attack",   
        "size": "large",
        "source": {
            "type": "kaggle",
            "slug": "ucimachinelearning/unsw-nb15-dataset",
            "file_path": "UNSW-NB15_Dataset2.csv",
            "requires_auth": True,
    },
},

    "cic_ids2017": {
        "name": "CIC-IDS2017",
        "domain": "cybersecurity",
        "task": "multiclass",
        "target": "Label",
        "size": "very_large",
        "source": {
            "type": "manual",
            "provider": "Canadian Institute for Cybersecurity",
            "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
            "expected_files": ["*.csv"],
        },
        "notes": "Very large dataset; manual download is recommended.",
    },

    "higgs": {
        "name": "HIGGS",
        "domain": "particle_physics",
        "task": "binary",
        "target": "class",
        "size": "very_large",
        "source": {
            "type": "manual",
            "provider": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/HIGGS",
            "expected_files": ["HIGGS.csv.gz"],
        },
        "notes": "Very large dataset; manual download is recommended.",
    },
}