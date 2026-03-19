import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

def train_deepsurv(df, duration_col="Follow Up Data", event_col="Total mortality", test_size=0.2, random_state=42):
    T = df[duration_col]
    E = df[event_col]
    X = df.drop(columns=[duration_col, event_col])

    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X, T, E, test_size=test_size, random_state=random_state
    )

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    T_train_tensor = torch.tensor(T_train.values, dtype=torch.float32)
    E_train_tensor = torch.tensor(E_train.values, dtype=torch.float32)
    T_test_tensor = torch.tensor(T_test.values, dtype=torch.float32)
    E_test_tensor = torch.tensor(E_test.values, dtype=torch.float32)

    in_features = X_train.shape[1]
    num_nodes = [32, 32]
    dropout = 0.1

    net = tt.practical.MLPVanilla(in_features, num_nodes, 1, batch_norm=True, dropout=dropout, activation=nn.ReLU)
    
    optimizer = tt.optim.AdamWR(lr=0.01)
    model = CoxPH(net, optimizer)

    epochs = 100
    batch_size = 128

    model.fit(X_train_tensor, (T_train_tensor, E_train_tensor), batch_size, epochs, verbose=True)
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(X_test_tensor)

    ev = EvalSurv(surv, T_test.values, E_test.values, censor_surv='km')
    c_index = ev.concordance_td()
    print("DeepSurv C-index:", c_index)

    return model, surv, ev
