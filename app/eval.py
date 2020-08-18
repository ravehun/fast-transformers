import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def nll_lognorm(mu, ln_sigma, target, eps=1e-6):
    return ln_sigma + (((target + eps).log() - mu) / ln_sigma.exp()) ** 2 / 2


def run(output_format, model, loader):
    model.cuda()
    model.eval()

    for i, z in tqdm.tqdm(enumerate(loader)):
        if i == 1:
            x, y, meta = z
    meta["stock_id"] = meta["stock_id"].cuda()
    meta['days_off'] = meta['days_off'].cuda()

    df = get_result(model, x, y, meta)
    stock_name = meta["stock_name"][0]

    df.to_parquet(output_format.format(stock_name=stock_name))


def get_result(model, x, y, meta):
    res = model.pred_with_attention(x.cuda(), meta)
    with torch.no_grad():
        loss = nll_lognorm(res['mu'], res['ln_sigma'], y.cuda())

    df = pd.DataFrame(data={
        "mu": res["mu"].exp().detach().cpu().reshape(-1),
        "sigma": res["ln_sigma"].exp().detach().cpu().reshape(-1),
        "y": y.reshape(-1),
        "x": x[0, :, 3],
        "loss": loss.detach().cpu().exp().reshape(-1)
    })

    return df


def do_plot(df, meta):
    fig, (ax1) = plt.subplots(1, 1, figsize=(30, 10))
    ax1.set_title(meta['stock_name'])
    ax1.set(xlim=(0, 1300))
    ax2 = ax1.twinx()
    ax2.set(ylim=(0, 1))

    def plot(x, ax, func, **kwargs):
        if type(x) == torch.Tensor:
            closed = x.numpy()
        if type(x) == pd.Series:
            closed = x.values
        func(x=np.arange(closed.shape[0]), y=closed, ax=ax, **kwargs)
        # plot(confidence,ax1)

    plot(df.x, ax1, sns.lineplot, color='red')
    plot(df.y, ax1, sns.lineplot)
    plot(df.mu, ax1, sns.scatterplot)
    plot(df.mmu_5, ax1, sns.lineplot)
    plot(df.loss, ax2, sns.lineplot)
    plot(df.mloss_10, ax2, sns.lineplot)

    plot(df.mx_20, ax1, sns.lineplot)
    plot(df.mx_60, ax1, sns.lineplot)
    plot(df.mx_120, ax1, sns.lineplot)

    # plot(df.sigma, ax2,sns.lineplot)
