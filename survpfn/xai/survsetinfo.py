
# Available: 'hdfail', 'e1684', 'phpl04K8a', 'prostate', 'cancer', 'DBCD', 'LeukSurv', 'Dialysis', 'Melanoma', 'd.oropha.rec', 'pharmacoSmoking', 'zinc', 'nki70', 'burn', 'breast', 'stagec', 'ovarian', 'actg', 'rhc', 'rdata', 'epileptic', 'vlbw', 'grace', 'TRACE', 'whas500', 'dataDIVAT1', 'Bergamaschi', 'AML_Bull', 'prostateSurvival', 'nwtco', 'divorce', 'dataDIVAT3', 'uis', 'colon', 'glioma', 'follic', 'mgus', 'Aids2', 'Z243', 'veteran', 'chop', 'wpbc', 'ova', 'micro.censure', 'MCLcleaned', 'cost', 'csl', 'heart', 'gse1992', 'smarto', 'NSBCD', 'retinopathy', 'support2', 'pbc', 'Pbc3', 'oldmort', 'UnempDur', 'cgd', 'acath', 'scania', 'Rossi', 'GBSG2', 'heartvalve', 'gse3143', 'Unemployment', 'gse4335', 'rott2', 'DLBCL', 'dataOvarian1', 'aids', 'diabetes', 'vdv', 'hepatoCellular', 'flchain', 'Framingham', 'FRTCS', 'dataDIVAT2
try:
    from SurvSet.data import SurvLoader
except ImportError as exc:
    raise ImportError(
        "SurvSet is not installed.  Run: uv add SurvSet"
    ) from exc

loader = SurvLoader()
SURVSET_BENCHMARK = loader.df_ds["ds"].tolist()

for ds_name in SURVSET_BENCHMARK:
    result = loader.load_dataset(ds_name=ds_name)
    df_raw = result["df"]
    ref = result["ref"]
    print(ds_name, ref)
