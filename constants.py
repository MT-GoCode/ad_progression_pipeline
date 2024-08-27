RANDOM_SEED = 42

TOTAL_VISITS = 7

cat_demo_feats = ['NACCNIHR','PRIMLANG','SEX','HISPANIC']
ord_demo_feats = ['MARISTAT','NACCLIVS','INDEPEND','RESIDENC']
num_demo_feats = ['NACCAGE','EDUC']

cat_phist_feats = ['TOBAC30', 'TOBAC100','NACCTBI','DEP2YRS', 'DEPOTHR',
        'ANYMEDS','NACCAAAS', 'NACCAANX', 'NACCAC', 'NACCACEI', 
        'NACCADEP', 'NACCAHTN', 'NACCANGI', 'NACCAPSY',
        'NACCBETA', 'NACCCCBS', 'NACCDBMD', 'NACCDIUR', 'NACCEMD',
        'NACCEPMD', 'NACCHTNC', 'NACCLIPL', 'NACCNSD', 'NACCPDMD',
        'NACCVASD']
ord_phist_feats = ['NACCAMD','PACKSPER','CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CBTIA',
        'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE','SEIZURES','NCOTHR',
        'DIABETES','HYPERTEN', 'HYPERCHO', 'B12DEF','THYROID', 'INCONTU', 
        'INCONTF','ALCOHOL', 'ABUSOTHR','PSYCDIS']
num_phist_feats = ['SMOKYRS','NACCSTYR','NACCTIYR']

cat_fhist_feats = ['NACCFADM', 'NACCFFTD']
ord_fhist_feats = ['NACCFAM', 'NACCMOM', 'NACCDAD']

cat_phys_feats = ['NACCNREX','FOCLSYM','FOCLSIGN']
ord_phys_feats = ['DECSUB','VISION', 'VISCORR','VISWCORR','HEARING', 'HEARAID', 'HEARWAID'] # 'APOERISK'
num_phys_feats = ['HEIGHT', 'WEIGHT','BPSYS', 'BPDIAS', 'HRATE','NACCBMI']

ord_gds_feats = ['NOGDS', # -4 = NaN - binary
        'SATIS', 'DROPACT', 'EMPTY', 'BORED', 'SPIRITS', 'AFRAID',
        'HAPPY', 'HELPLESS', 'STAYHOME', 'MEMPROB', 'WONDRFUL', 'WRTHLESS',
        'ENERGY', 'HOPELESS', 'BETTER', # recode 9 as -1, -4 = NaN - ordinal
        'NACCGDS'] # 88 or -4 = NaN - ordinal
ord_faq_feats = ['BILLS', 'TAXES','SHOPPING', 'GAMES', 'STOVE', 
        'MEALPREP', 'EVENTS', 'PAYATTN','REMDATES', 'TRAVEL'] # recode 8 as -1, 9 or -4 = NaN
# Decide to drop the binary variables in NPI
ord_npi_feats = ['DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 'ANXSEV',
            'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
            'APPSEV',]

ord_np_feats = ['MMSEORDA','MMSEORLO']

label_feat = ['NACCUDSD']
cdr_feats = ['MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 'HOMEHOBB', 
                    'PERSCARE', 'CDRSUM', 'CDRGLOB']

# combined
CATEGORICAL_FEATURES = cat_demo_feats + cat_phist_feats + cat_fhist_feats + cat_phys_feats
num_np_feats = ['NACCMMSE','MEMUNITS','DIGIF', 'DIGIFLEN', 'DIGIB', 'DIGIBLEN',
                'ANIMALS', 'VEG','BOSTON', 'TRAILA', 'TRAILB']
ORDINAL_FEATURES = list(ord_demo_feats + ord_phist_feats + ord_fhist_feats + ord_phys_feats + 
                ord_npi_feats + ord_gds_feats + ord_faq_feats + ord_np_feats)
NUMERICAL_FEATURES = num_demo_feats + num_phist_feats + num_phys_feats + num_np_feats