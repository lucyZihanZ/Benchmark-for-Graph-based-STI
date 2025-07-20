from tsl.datasets.prototypes import DatetimeDataset
import os
import numpy as np
import pandas as pd
import sys
# from utils_files.utils import sample_mask

class WaterQuality(DatetimeDataset):
    """Custom water quality dataset using SSC_pooled.csv and SSC_sites_flow_direction.csv."""
    similarity_options = {'distance'}

    def __init__(self, impute_zeros=True, root='ssc', freq='1H'):
        self.root = root
        df, mask = self.load(impute_zeros)
        super().__init__(
            target=df,
            mask=mask,
            freq=freq,
            similarity_score="distance",
            temporal_aggregation="nearest",
            name="WaterQuality"
        )
        # self.add_covariate('dist', dist, pattern='n n')

    def load(self, impute_zeros=True):
        # data_path = os.path.join(self.root, 'ssc/SSC_pooled.csv')
        # adj_path = os.path.join(self.root, 'SSC_sites_flow_direction.csv')
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the CSV file relative to the script location
        csv_path = os.path.join(script_dir, 'ssc', 'SSC_pooled.csv')

        # Now use the absolute path
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        date_range = pd.date_range(df.index[0], df.index[-1], freq='1H')
        df = df.reindex(index=date_range)

        values = df.values
        mask = ~np.isnan(values)
        if impute_zeros:
            mask = mask & (values != 0)
            df = df.replace(0., np.nan).fillna(method='ffill').fillna(method='bfill')

        # adj = pd.read_csv(adj_path, index_col=0).values
        return df.astype('float32'), mask.astype('uint8')

    # def compute_similarity(self, **kwargs):
        # if method != 'distance':
        #     raise NotImplementedError(f"Similarity method {method} not implemented.")
        # finite_dist = self.dist.reshape(-1)
        # finite_dist = finite_dist[~np.isinf(finite_dist)]
        # sigma = finite_dist.std()
        # return np.exp(-(self.dist ** 2) / (2 * sigma ** 2))
    # def compute_similarity(self, threshold=0.0, force_symmetric=False, sparse=False):
    #     script_dir = os.path.dirname(os.path.abspath(__file__))
    #     csv_path = os.path.join(script_dir, 'ssc', 'SSC_sites_flow_direction.csv')
    #     try:
    #         adj = pd.read_csv(csv_path) # for imputeformer
    #     except:
    #         adj = pd.read_csv(csv_path).values # for spin (as SPIN forcely transformed into the matrix.)
    #     # adj = pd.read_csv('ssc/SSC_sites_flow_direction.csv')
    #     # if threshold > 0:
    #     #     adj[adj < threshold] = 0.
    #     if force_symmetric:
    #         adj = np.maximum(adj, adj.T)
    #     if sparse:
    #         import scipy.sparse as sps
    #         adj = sps.coo_matrix(adj)
    def compute_similarity(self, threshold=0.0, force_symmetric=False, sparse=False):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        npy_path = os.path.join(script_dir, 'ssc', 'adj.npy')

        adj = np.load(npy_path)  # 直接加载 numpy 文件，非常稳定

        # if threshold > 0:
        #     adj[adj < threshold] = 0.
        if force_symmetric:
            adj = np.maximum(adj, adj.T)
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)

        print(type(adj))
        return adj
# if __name__ == "__main__":
# import pandas as pd
# import numpy as np
# import os
# csv_path = os.path.join('ssc', 'SSC_sites_flow_direction.csv')
# adj = pd.read_csv(csv_path, index_col=0).values
# np.save('ssc/adj.npy', adj)

    
