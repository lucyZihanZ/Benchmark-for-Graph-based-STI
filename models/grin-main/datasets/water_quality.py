# import os
# import numpy as np
# import pandas as pd
# # from utils_files.utils import masking import sample_mask  # make sure this exists
# from utils_files.utils import sample_mask
# # from base import PandasDataset  # assumed from your repo

# class WaterQuality():
#     def __init__(self, impute_zeros=True, freq='1H'):
#         df, adj, mask = self.load(impute_zeros=impute_zeros)
#         self.adj = adj  # directly use the adjacency matrix
#         super().__init__(dataframe=df, u=None, mask=mask, name='water', freq=freq, aggr='nearest')

#     def load(self, impute_zeros=True):
#         file_path = os.path.join('ssc', 'SSC_pooled.csv')
#         df = pd.read_csv(file_path)
        
#         # Reindex time to ensure consistent temporal spacing
#         datetime_idx = sorted(df.index)
#         date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1H')
#         df = df.reindex(index=date_range)
        
#         # Generate mask
#         mask = ~np.isnan(df.values)
#         if impute_zeros:
#             mask = mask * (df.values != 0.).astype('uint8')
#             df = df.replace(to_replace=0., method='ffill')
#         else:
#             mask = None
        
#         # Load your precomputed adjacency matrix from CSV
#         adj = self.load_adjacency_matrix()

#         return df.astype('float32'), adj.astype('float32'), mask

#     def load_adjacency_matrix(self):
#         """
#         Load your edge list or adjacency matrix from pooled.csv.
#         Expecting a square matrix with no headers.
#         """
#         path = os.path.join('ssc', 'SSC_sites_flow_direction.csv')
#         adj = pd.read_csv(path, header=None).values  # shape [N, N]
#         return adj

#     def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
#         adj = self.adj.copy()
#         adj[adj < thr] = 0.
#         if force_symmetric:
#             adj = np.maximum.reduce([adj, adj.T])
#         if sparse:
#             import scipy.sparse as sps
#             adj = sps.coo_matrix(adj)
#         return adj

#     @property
#     def mask(self):
#         return self._mask


# class MissingValuesWaterQuality(WaterQuality):
#     SEED = 12345

#     def __init__(self, p_fault=0.0015, p_noise=0.05):
#         super(MissingValuesWaterQuality, self).__init__(impute_zeros=True)
#         self.rng = np.random.default_rng(self.SEED)
#         self.p_fault = p_fault
#         self.p_noise = p_noise
#         eval_mask = sample_mask(self.numpy().shape,
#                                 p=p_fault,
#                                 p_noise=p_noise,
#                                 min_seq=12,
#                                 max_seq=12 * 4,
#                                 rng=self.rng)
#         self.eval_mask = (eval_mask & self.mask).astype('uint8')

#     @property
#     def training_mask(self):
#         return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

#     def splitter(self, dataset, val_len=0, test_len=0, window=0):
#         idx = np.arange(len(dataset))
#         if test_len < 1:
#             test_len = int(test_len * len(idx))
#         if val_len < 1:
#             val_len = int(val_len * (len(idx) - test_len))
#         test_start = len(idx) - test_len
#         val_start = test_start - val_len
#         return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]
import os
import numpy as np
import pandas as pd
from utils_files.utils import sample_mask

class WaterQuality():
    def __init__(self, impute_zeros=True, freq='1H'):
        df, adj, mask = self.load(impute_zeros=impute_zeros)

        # Assign loaded components to internal attributes.
        # Use _dataframe, _adj, _mask to avoid conflicts with @property methods
        # or if you intend to add properties later.
        self._dataframe = df
        self._adj = adj
        self._mask = mask # Store the mask in an internal attribute
        self.name = 'water'
        self.freq = freq
        # No super().__init__() as you are not inheriting from another specific dataset class

    def load(self, impute_zeros=True):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_pooled_csv = os.path.join(current_file_dir, 'ssc', 'SSC_pooled.csv')

        if not os.path.exists(file_path_pooled_csv):
            raise FileNotFoundError(f"Data file not found at: {file_path_pooled_csv}")

        # Assuming the first column is the datetime index and needs parsing
        df = pd.read_csv(file_path_pooled_csv, index_col=0, parse_dates=True)
        self.df = df
        print(f"DEBUG: SSC_pooled.csv data shape after index_col=0: {df.shape}") # Should be (time_steps, actual_nodes)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not convert DataFrame index to datetime for reindexing. Error: {e}")

        datetime_idx = sorted(df.index)
        if not datetime_idx:
            raise ValueError("DataFrame index is empty after loading, cannot create date_range.")

        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1H')
        df = df.reindex(index=date_range)

        data_values = df.values
        mask = ~np.isnan(data_values)

        if impute_zeros:
            mask = mask * (data_values != 0.).astype('uint8')
            df = df.replace(to_replace=0., method='ffill').fillna(method='ffill').fillna(method='bfill')
        else:
            mask = None

        adj = self.load_adjacency_matrix(current_file_dir=current_file_dir)

        return df.astype('float32'), adj.astype('float32'), mask

    def load_adjacency_matrix(self, current_file_dir):
        adj_file_path = os.path.join(current_file_dir, 'ssc', 'SSC_sites_flow_direction.csv')
        if not os.path.exists(adj_file_path):
            raise FileNotFoundError(f"Adjacency matrix file not found at: {adj_file_path}")
        adj = pd.read_csv(adj_file_path, index_col = 0, parse_dates=True).values
        print(f"DEBUG: SSC_adj.csv data shape after index_col=0: {adj.shape}") # Should be (time_steps, actual_nodes)
        return adj

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
        adj = self.adj.copy()
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        # This property now safely returns the internal _mask attribute.
        # It will not cause recursion or AttributeError when self._mask is set.
        return self._mask if hasattr(self, '_mask') else None # Added hasattr for robustness

    # Also add properties for dataframe and adj for consistent access if needed
    @property
    def dataframe(self):
        return self._dataframe if hasattr(self, '_dataframe') else None

    @property
    def adj(self):
        return self._adj if hasattr(self, '_adj') else None


# -----------------------------------------------------------------------------
# MissingValuesWaterQuality Class
# -----------------------------------------------------------------------------

class MissingValuesWaterQuality(WaterQuality):
    SEED = 12345

    def __init__(self, p_fault=0.0015, p_noise=0.05):
        # Call WaterQuality's __init__ to set up _dataframe, _adj, _mask
        super(MissingValuesWaterQuality, self).__init__(impute_zeros=True)

        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        
        # Access the mask via the @property (self.mask) or direct internal attribute (self._mask)
        # Using self.mask will call the getter, ensuring correct behavior.
        base_mask_for_sampling = self.mask
        if base_mask_for_sampling is None:
            # Fallback if self.mask from WaterQuality's load was None (e.g., impute_zeros=False)
            # You might need to adjust this fallback based on whether self.dataframe is expected to exist here.
            base_mask_for_sampling = ~np.isnan(self.dataframe.values) if self.dataframe is not None else np.ones(self.shape, dtype=bool) # Assuming self.shape exists or define it.
            
        # Ensure that self.dataframe exists and has a .values attribute for .shape
        if self.dataframe is None:
             raise ValueError("DataFrame not initialized in WaterQuality. Make sure data loads correctly.")

        eval_mask = sample_mask(self.dataframe.shape, # Pass the shape of the data for mask generation
                                p=p_fault,
                                p_noise=p_noise,
                                min_seq=12,
                                max_seq=12 * 4,
                                rng=self.rng)
        
        self.eval_mask = (eval_mask & base_mask_for_sampling).astype('uint8')

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]