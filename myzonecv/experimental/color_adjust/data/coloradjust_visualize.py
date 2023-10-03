import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw_histograms(data_list,
                    file_path,
                    width=800,
                    height=600,
                    title=None,
                    subtitles=None,
                    binsize=None,
                    nbins=25,
                    xrange=[0, 1],
                    horizontal_spacing=0.01,
                    margin={'l': 20, 'r': 20, 't': 80, 'b': 20}):
    cols = len(data_list)
    if subtitles:
        assert len(subtitles) == cols
        subtitles = [f'{t} ({d.mean():.5f}+/-{d.std():.5f})' for d, t in zip(data_list, subtitles)]
    comparison_fig = make_subplots(rows=1, cols=cols,
                                   subplot_titles=subtitles, shared_yaxes=True, horizontal_spacing=horizontal_spacing)

    for i, data in enumerate(data_list):
        if binsize is not None:
            nbinsx = int(np.ceil((data.max() - data.min()) / binsize))
        else:
            nbinsx = nbins
        comparison_fig.append_trace(go.Histogram(x=data, nbinsx=nbinsx), row=1, col=i + 1)

    width *= cols
    comparison_fig.update_layout(width=width, height=height, title_text=title, margin=margin)
    comparison_fig.update_xaxes(range=xrange)
    comparison_fig.write_image(file_path)
