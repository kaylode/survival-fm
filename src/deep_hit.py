import torch
from torch import nn
from sklearn.model_selection import train_test_split
from pycox.models import DeepHitSingle
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

def train_deephit_competing_risks(df, duration_col="Follow Up Data", event_col="death", test_size=0.2, random_state=42):
    T = df[duration_col]
    E = df[event_col]
    X = df.drop(columns=[duration_col, event_col])

    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X, T, E, test_size=test_size, random_state=random_state
    )

    X_train_np = X_train.values.astype('float32')
    X_test_np = X_test.values.astype('float32')

    num_durations = 100
    labtrans = LabTransDiscreteTime(num_durations)

    y_train = labtrans.fit_transform(T_train.values, E_train.values)
    y_test = labtrans.transform(T_test.values, E_test.values)

    out_features = labtrans.out_features
    print("Network output nodes needed:", out_features)

    in_features = X_train.shape[1]
    num_nodes = [64, 32]
    dropout = 0.1

    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features=out_features,
        batch_norm=True, dropout=dropout, activation=nn.ReLU
    )

    optimizer = tt.optim.AdamWR(lr=0.01)
    model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)

    epochs = 50
    batch_size = 128

    model.fit(X_train_np, y_train, batch_size=batch_size, epochs=epochs, verbose=True)

    surv = model.predict_surv_df(X_test_np)

    ev = EvalSurv(surv, T_test.values, E_test.values, censor_surv='km')
    c_index = ev.concordance_td()
    print("DeepHit C-index:", c_index)

    return model, surv, ev
