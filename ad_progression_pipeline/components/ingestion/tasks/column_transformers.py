# # Utility functions for data cleaning
# # Yueqi Ren, 2023-10-17

# #############################################


# #############################################

# def get_feature_types(df,tier2=True):
#     """ Obtain lists of features that fall into different data types.
#         Args:
#             df: pandas dataframe
#             tier2: boolean, whether to include tier2 features (default is True),
#                 can leave as is without change
#     """
#     # one hot encode, without dropping categories

#     cat_phist_feats = ["TOBAC30", "TOBAC100","NACCTBI","DEP2YRS", "DEPOTHR",
#             "ANYMEDS","NACCAAAS", "NACCAANX", "NACCAC", "NACCACEI",
#             "NACCADEP", "NACCAHTN", "NACCANGI", "NACCAPSY",
#             "NACCBETA", "NACCCCBS", "NACCDBMD", "NACCDIUR", "NACCEMD",
#             "NACCEPMD", "NACCHTNC", "NACCLIPL", "NACCNSD", "NACCPDMD",
#             "NACCVASD"]
#     ord_phist_feats = ["NACCAMD","PACKSPER","CVHATT", "CVAFIB", "CVANGIO", "CVBYPASS", "CBTIA",
#             "CVPACE", "CVCHF", "CVOTHR", "CBSTROKE","SEIZURES","NCOTHR",
#             "DIABETES","HYPERTEN", "HYPERCHO", "B12DEF","THYROID", "INCONTU",
#             "INCONTF","ALCOHOL", "ABUSOTHR","PSYCDIS"]


#     ord_gds_feats = ["NOGDS", # -4 = NaN - binary
#             "SATIS", "DROPACT", "EMPTY", "BORED", "SPIRITS", "AFRAID",
#             "HAPPY", "HELPLESS", "STAYHOME", "MEMPROB", "WONDRFUL", "WRTHLESS",
#             "NACCGDS"] # 88 or -4 = NaN - ordinal
#     ord_faq_feats = ["BILLS", "TAXES","SHOPPING", "GAMES", "STOVE",
#             "MEALPREP", "EVENTS", "PAYATTN","REMDATES", "TRAVEL"] # recode 8 as -1, 9 or -4 = NaN
#     # Decide to drop the binary variables in NPI
#     ord_npi_feats = ["DELSEV", "HALLSEV", "AGITSEV", "DEPDSEV", "ANXSEV",
#                 "ELATSEV", "APASEV", "DISNSEV", "IRRSEV", "MOTSEV", "NITESEV",
#                 "APPSEV"]


#     # CDR features are never missing and are all ordinal
#     cdr_feats = ["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB",
#                         "PERSCARE", "CDRSUM", "CDRGLOB"]

#     # combined
#     if tier2:
#         num_np_feats = ["NACCMMSE","MEMUNITS","DIGIF", "DIGIFLEN", "DIGIB", "DIGIBLEN",
#                     "ANIMALS", "VEG","BOSTON", "TRAILA", "TRAILB"]
#     ord_feats = list(ord_demo_feats + ord_phist_feats + ord_fhist_feats + ord_phys_feats +
#                     ord_npi_feats + ord_gds_feats + ord_faq_feats + ord_np_feats)

#     # check that the features are in the dataframe


# def encode_feature_types(X_data,tier2=True):
#     """ Encode features by data type for later processing.
#         Args:
#             X_data: pandas dataframe, n x p
#             tier2: boolean, whether to include tier2 features (default is True),
#                 can leave as is without change
#         Notes:
#             All the imputation steps are not used here but gives you an idea
#             of how you can incorporate it in the same set up later on
#             (imputation is done in the pipeline AFTER train/test split).
#             If you have CDR or diagnosis features, uncomment the final part of the function.
#     """
#     # Use specific transformers for each type of data

#     # Obtain the ordinal-encoded features to get the names of the columns
#        ],
#             # If your data frame includes other features, uncomment the line below
#         ],


# def split_data(data,test_percent=0.2,random_seed=None):
#     """ Split the data into training and testing sets.
#         Args:
#             data: pandas dataframe, n x p of all features (including the clinical diagnosis label)
#             test_percent: float, percentage of data to use for testing
#             random_seed: int or None (default), to ensure reproducibility for splits if given a specific seed
#     """


# def match_invariant_feature_names(transformed_features):
#     """ Due to one-hot-encoding, make sure to match feature names
#         Args:
#             transformed_features: list of feature names after encoding
#     """
#     # Find which features are time invariant
#     for feat in transformed_features:
#         if feat.split("_")[0] in invariant_feats:


# def impute_invariant_features(X_train, X_test):
#     """ Impute all invariant features using training data.
#         Args:
#             X_train: pandas dataframe, n x p of training data
#             X_test: pandas dataframe, n x p of testing data
#         Notes:
#             You may need to change the file directory for the feature_details file.
#     """
#     # Find which features are time invariant
#     # Impute invariant features
#     for feat in invariant_feats:
#         # filling in using the mode ensures that all missing instances receive the same value
#         # feel free to edit this step to use other types of imputation methods
#         # while ensuring that all values of the feature are the same per subject across visits


# def imputer_by_feature_type(X_data,tier2=True):
#     """ Impute features by data type using training data.
#         Args:
#             X_data: pandas dataframe, n x p of training data
#             tier2: boolean, whether to include tier2 features (default is True),
#                 can leave as is without change
#         Notes:
#             All the training data should have already be encoded by data type prior to this step.
#             If you have CDR or diagnosis features, uncomment the final part of the function.
#     """
#     # Use specific transformers for each type of data

#         ],
#         ],
#        ],
#             # If your data frame includes other features, uncomment the line below
#         ],


# #############################################

# if __name__ == "__main__":
#     # Perform the data cleaning steps as outlined in the Data Cleaning document

#     # Load the data
#     # Merge the training and testing data for data cleaning
#     # Select the features to use
#     # Split the data into training and testing sets
#     # Impute the invariant features
#     # Impute the remaining features
#     # Save the data
