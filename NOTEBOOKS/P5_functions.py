"""Decorator to time functions
Call using following model when creating the function to be timed:
    @timing
    def function(a):
        pass
"""

import time
from functools import wraps

def timing(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print('function:%r took: %2.2f sec' % (f.__name__,  end - start))
        return result
    return wrapper

# Printing total nb and percentage of null

import pandas as pd

def print_null_pct(df):
    tot_null = df.isna().sum().sum()
    print('nb of null: ', tot_null, '\npct of null: ',
        '{:.1f}'.format(tot_null*100/(df.shape[0]*df.shape[1])))

# Displaying number of missing values per column

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_export_missing(df, cols, n_file, title,
                        shorten_label=False, figsize=(12,8), save_enabled=False):
    with plt.style.context('default'):
        fig, axs = plt.subplots(2,1)
        msno.matrix(df[cols] , sparkline=False,
                    fontsize=11, ax=axs[0])
        msno.bar(df[cols], ax=axs[1], fontsize=11)
        if shorten_label:
            for ax in axs:
                lab = [item.get_text() for item in ax.get_xticklabels()]
                short_lab = [s[:7]+'...'+s[-7:] if len(s)>14 else s for s in lab]
                ax.axes.set_xticklabels(short_lab)
    fig.set_size_inches(figsize)
    [ax.grid() for ax in axs.flatten()];
    [sns.despine(ax=ax, right=False, left=False,top=False, bottom=False)\
                                        for ax in axs.flatten()];
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.suptitle(title, fontweight='bold', fontsize=14)
    if not os.path.exists(os.getcwd()+'/FIG'):
        os.makedirs('FIG')
    if save_enabled: plt.savefig(os.getcwd()+'/FIG/'+n_file, dpi=400);
    plt.show()

''' Function that changes the order of the columns: gathers the columns that contains
one or more of the strings listed in 'strings' and puts them at the beginning ('first')
or at the end ('last') '''

import pandas as pd

def move_cat_containing(my_index, strings, order='last'):
	idx_sel = []
	if order == 'last':
	    index = pd.Index(my_index)
	elif order == 'first':
	    index = pd.Index(my_index[::-1])
	else:
	    print("--- WARNING : index unchanged.\n -- Wrong order passed. Pass 'first' or 'last'")
	    return my_index
	for s in strings:
	    idx_sel += [i for i,x in enumerate(index) if x in index[index.str.contains(s)]]
	to_move = index[idx_sel]
	mod_index = index.drop(to_move)
	for c in to_move:
	    mod_index = mod_index.insert(len(mod_index),c)
	return mod_index if order=='last' else mod_index[::-1]

''' Function that computes the mean for Timedeltas'''

import pandas as pd
from datetime import timedelta

def mean_tdeltas(ser_td) :
    return sum(ser_td, timedelta(0)) / len(ser_td)


''' Builds a customizable column_transformer which parameters can be optimized in a GridSearchCV
CATEGORICAL : three differents startegies for 3 different types of
categorical variables:
- low cardinality: customizable strategy (strat_low_card)
- high cardinality: customizable strategy (strat_high_card)
- boolean or equivalent (2 categories): ordinal
QUANTITATIVE (remainder): 
- StandardScaler

-> EXAMPLE (to use apart from gscv):
cust_enc = CustTransformer(thresh_card=12,
                       strat_binary = 'ord',
                       strat_low_card = 'ohe',
                       strat_high_card = 'loo',
                       strat_quant = 'stand')
cust_enc.fit(X_tr, y1_tr)
cust_enc.transform(X_tr).shape, X_tr.shape

-> EXAMPLE (to fetch names of the modified dataframe):
small_df = df[['Outlier', 'Neighborhood', 'CertifiedPreviousYear',
               'NumberofFloors','ExtsurfVolRatio']]
# small_df.head(2)
cust_trans = CustTransformer()
cust_trans.fit(small_df)
df_enc = cust_trans.transform(small_df)
cust_trans.get_feature_names(small_df)

'''
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import *
import numpy as np
import pandas as pd


class CustTransformer(BaseEstimator):

    def __init__(self, thresh_card=12,
                 strat_binary='ord', strat_low_card='ohe',
                 strat_high_card='bin', strat_quant='stand'):
        self.thresh_card = thresh_card
        self.strat_binary = strat_binary
        self.strat_low_card = strat_low_card
        self.strat_high_card = strat_high_card
        self.strat_quant = strat_quant
        self.dict_enc_strat = {'binary': strat_binary,
                               'low_card': strat_low_card,
                               'high_card': strat_high_card,
                               'numeric': strat_quant}

    def d_type_col(self, X):
        bin_cols = (X.nunique()[X.nunique() <= 2].index)
        X_C_cols = X.select_dtypes(include=['object', 'category'])
        C_l_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique() \
                .between(3, self.thresh_card)].index
        C_h_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique() > self.thresh_card].index
        Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns \
                  if c not in bin_cols]
        d_t = {'binary': bin_cols,
               'low_card': C_l_card_cols,
               'high_card': C_h_card_cols,
               'numeric': Q_cols}
        d_t = {k: v for k, v in d_t.items() if len(v)}
        # print(d_t)
        return d_t

    def get_feature_names(self, X, y=None):
        if self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names() + self.num_cols
        elif self.has_num and not self.has_cat:
            cols = self.num_cols
        elif not self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names()
        else:
            cols = None
        return cols

    def fit(self, X, y=None):
        # Dictionary to translate strategies
        d_enc = {'ohe': ce.OneHotEncoder(),
                 'hash': ce.HashingEncoder(),
                 'ord': ce.OrdinalEncoder(),
                 'loo': ce.LeaveOneOutEncoder(),
                 'bin': ce.BinaryEncoder(),
                 'stand': StandardScaler(),
                 'minmax': MinMaxScaler(),
                 'maxabs': MaxAbsScaler(),
                 'robust': RobustScaler(quantile_range=(25, 75)),
                 'norm': Normalizer(),
                 'quant_uni': QuantileTransformer(output_distribution='uniform'),
                 'quant_norm': QuantileTransformer(output_distribution='normal'),
                 'boxcox': PowerTransformer(method='box-cox'),
                 'yeo': PowerTransformer(method='yeo-johnson'),
                 'log': FunctionTransformer(func=lambda x: np.log1p(x),
                                            inverse_func=lambda x: np.expm1(x)),
                 'none': FunctionTransformer(func=lambda x: x,
                                             inverse_func=lambda x: x),
                 }

        # # dictionnaire liste des transfo categorielles EXISTANTES
        d_t = self.d_type_col(X)
        # numerics
        self.has_num = ('numeric' in d_t.keys())
        # categoricals
        self.has_cat = len([s for s in d_t.keys() if s in ['binary', 'low_card', 'high_card']]) > 0
        if self.has_cat:
            list_trans = []  # dictionnaire des transfo categorielles EXISTANTES
            for k, v in d_t.items():
                if k != 'numeric':
                    list_trans.append((k, d_enc[self.dict_enc_strat[k]], v))

            self.cat_cols = []  # liste des colonnes catégorielles à transformer
            for k, v in self.d_type_col(X).items():
                if k != 'numeric': self.cat_cols += (list(v))

            self.ct_cat = ColumnTransformer(list_trans)
            self.cat_trans = Pipeline([("categ", self.ct_cat)])

        if self.has_num:
            self.num_trans = Pipeline([("numeric", d_enc[self.strat_quant])])
            self.num_cols = d_t['numeric']

        if self.has_num and self.has_cat:
            self.column_trans = \
                ColumnTransformer([('cat', self.cat_trans, self.cat_cols),
                                   ('num', self.num_trans, self.num_cols)])
        elif self.has_num and not self.has_cat:
            self.column_trans = \
                ColumnTransformer([('num', self.num_trans, self.num_cols)])
        elif not self.has_num and self.has_cat:
            self.column_trans = ColumnTransformer([('cat', self.cat_trans, self.cat_cols)])
        else:
            print("The dataframe is empty : no transformation can be done")

        return self.column_trans.fit(X, y)

    # OLD VERSION WITHOUT NAME OF THE COLUMNS, OUTPUT AS A NP.ARRAY
    # def transform(self, X, y=None): 
    #     return  self.column_trans.transform(X)
    # def fit_transform(self, X, y=None):
    #     if y is None:
    #         self.fit(X)
    #         return self.column_trans.transform(X)
    #     else:
    #         self.fit(X, y)
    #         return self.column_trans.transform(X)

    def transform(self, X, y=None):  # to get a dataframe
        return pd.DataFrame(self.column_trans.transform(X),
                            index=X.index,
                            columns=self.get_feature_names(X, y))

    def fit_transform(self, X, y=None):
        if y is None:
            self.fit(X)
            return pd.DataFrame(self.column_trans.transform(X),
                                index=X.index,
                                columns=self.get_feature_names(X, y))
        else:
            self.fit(X, y)
            return pd.DataFrame(self.column_trans.transform(X, y),
                                index=X.index,
                                columns=self.get_feature_names(X, y))


# Plotting histograms of specified quantitative continuous columns of a dataframe and mean, median and mode values.

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms(df, cols, file_name=None, bins=30, figsize=(12,7), color = 'grey',
                    skip_outliers=False, thresh=3, layout=(3,3), save_enabled=False):

    fig = plt.figure(figsize=figsize)

    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)
        if skip_outliers:
            ser = df[c][np.abs(st.zscore(df[c]))<thresh]
        else:
            ser = df[c]
        ax.hist(ser,  bins=bins, color=color)
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(),  color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(), color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(), color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))
        
    plt.tight_layout(w_pad=0.5, h_pad=0.65)

    if save_enabled: plt.savefig(os.getcwd()+'/FIG/'+file_name, dpi=400);
    plt.show()

# Plotting bar plots of the main categorical columns

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_barplots(df, cols, file_name=None, figsize=(12,7), layout=(2,3), save_enabled=False):

    fig = plt.figure(figsize=figsize)
    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)
        ser = df[c].value_counts()
        n_cat = ser.shape[0]
        if n_cat>15:
            ser[0:15].plot.bar(color='grey',ec='k', ax=ax)
        else:
            ser.plot.bar(color='grey',ec='k',ax=ax)
        ax.set_title(c[0:17]+f' ({n_cat})', fontweight='bold')
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s)>7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_enabled: plt.savefig(os.getcwd()+'/FIG/'+file_name, dpi=400);
    plt.show()

# Plotting heatmap (2 options available, rectangle or triangle )

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_heatmap(corr, title, figsize=(8, 4), vmin=-1, vmax=1, center=0,
                 palette=sns.color_palette("coolwarm", 20), shape='rect',
                 fmt='.2f', robust=False, fig=None, ax=None):
    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    if shape == 'rect':
        mask = None
    elif shape == 'tri':
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        print('ERROR : this type of heatmap does not exist')

    palette = palette
    ax = sns.heatmap(corr, mask=mask, cmap=palette, vmin=vmin, vmax=vmax,
                     center=center, annot=True, annot_kws={"size": 10}, fmt=fmt,
                     square=False, linewidths=.5, linecolor='white',
                     cbar_kws={"shrink": .9, 'label': None}, robust=robust,
                     xticklabels=corr.columns, yticklabels=corr.index,
                     ax=ax)
    ax.tick_params(labelsize=10, top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontweight='bold', fontsize=12)


# Plotting explained variance ratio in scree plot

import pandas as pd
import matplotlib.pyplot as plt

def scree_plot(col_names, exp_var_rat, ylim=(0, 0.4), figsize=(8, 3)):
    plt.bar(x=col_names, height=exp_var_rat, color='grey')
    ax1 = plt.gca()
    ax1.set(ylim=ylim)
    ax2 = ax1.twinx()
    ax2.plot(exp_var_rat.cumsum(), 'ro-')
    ax2.set(ylim=(0, 1.1))
    ax1.set_ylabel('explained var. rat.')
    ax2.set_ylabel('cumulative explained var. rat.')

    for i, p in enumerate(ax1.patches):
        ax1.text(p.get_width() / 5 + p.get_x(), p.get_height() + p.get_y() + 0.01,
                 '{:.0f}%'.format(exp_var_rat[i] * 100),
                 fontsize=8, color='k')

    plt.gcf().set_size_inches(figsize)
    plt.title('Scree plot', fontweight='bold')


''' displays the lines on the first axis of a pca'''

def display_factorial_planes(X_proj, n_comp, pca, axis_ranks, labels=None,
                             width=16, alpha=1, n_cols=3, illus_var=None,
                             lab_on=True, size=10):
    n_rows = (n_comp+1)//n_cols
    fig = plt.figure(figsize=(width,n_rows*width/n_cols))
    # boucle sur chaque plan factoriel
    for i, (d1,d2) in (enumerate(axis_ranks)):
        if d2 < n_comp:
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            # points
            if illus_var is None:
                ax.scatter(X_proj[:, d1], X_proj[:, d2], alpha=alpha, s=size)
            else:
                illus_var = np.array(illus_var)
                for value in np.unique(illus_var):
                    sel = np.where(illus_var == value)
                    ax.scatter(X_proj[sel, d1], X_proj[sel, d2], 
                                alpha=alpha, label=value)
                ax.legend()
            # labels points
            if labels is not None and lab_on:
                for i,(x,y) in enumerate(X_proj[:,[d1,d2]]):
                    ax.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center')   
            # limites
            bound = np.max(np.abs(X_proj[:, [d1,d2]])) * 1.1
            ax.set(xlim=(-bound,bound), ylim=(-bound,bound))
            # lignes horizontales et verticales
            ax.plot([-100, 100], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-100, 100], color='grey', ls='--')
            # nom des axes, avec le pourcentage d'inertie expliqué
            ax.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            ax.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            ax.set_title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
    plt.tight_layout()


'''Aggregate all the orders of one unique customer
to get a database of customers
Uses a period of time on which aggregation is done.
If no min and max time given, aggregates on the whole df_order
'''

def create_agg_cust_df(df_orders, t_min=None, t_max=None): # t_min not rally useful
    
    df_orders_mod = df_orders.copy('deep')
    # Changes the type of data
    df_orders_mod[['order_purchase_timestamp', 'max_limit_ship_date']] =\
        df_orders_mod[['order_purchase_timestamp', 'max_limit_ship_date']]\
        .apply(lambda x: pd.to_datetime(x))
    df_orders_mod[['shipping_time', 'shipping_delay']] = \
        df_orders_mod[['shipping_time', 'shipping_delay']]\
        .apply(lambda x: pd.to_timedelta(x))
    df_orders_mod['cust_region'] = \
        df_orders_mod['cust_region'].astype('object')

    if t_min is None:
        t_min = df_orders_mod['order_purchase_timestamp'].min()
    if t_max is None:
        t_max = df_orders_mod['order_purchase_timestamp'].max()

    mask_time = df_orders_mod['order_purchase_timestamp'].between(t_min, t_max)
    df = df_orders_mod[mask_time].reset_index()

    def most_freq_cat(x): return x if x.size == 1 else x.value_counts().idxmax()

    # Dictionary for the aggregation of the main values and times
    agg_dict_1 = {
                # 'cust_zipcode': ('customer_zip_code_prefix', np.max),
                # 'cust_city': ('customer_city', np.max),
                # 'cust_state': ('customer_state', np.max),
                'cust_region': ('cust_region', np.max),
                'tot_nb_ord': ('order_id', np.size),
                'tot_nb_deliv_ord': ('delivered', np.sum),
                'time_since_last_purch': ('order_purchase_timestamp',
                                            lambda x: t_max - np.max(x)),
                'time_since_first_purch': ('order_purchase_timestamp',
                                            lambda x: t_max - np.min(x)),
                'mean_ship_time': ('shipping_time', mean_tdeltas),
                'mean_ship_delay': ('shipping_time', mean_tdeltas),
                'tot_nb_items': ('order_item_nb', np.sum),
                'mean_nb_items_per_ord': ('order_item_nb', np.mean),
                'mean_prod_descr_length': ('mean_prod_descr_length', np.mean),
                'mean_prod_vol_cm3': ('product_volume_cm3', np.mean),
                'mean_prod_wei_g': ('product_weight_g', np.mean),
                'mean_price_per_order': ('price', np.mean),
                'mean_freight_val_per_order': ('freight_value', np.mean),
                'mean_pay_value_per_order': ('payment_value', np.mean),
                'tot_price': ('price', np.sum),
                'tot_freight_val': ('freight_value', np.sum),
                'tot_pay_value': ('payment_value', np.sum),
                'mean_pay_install': ('payment_installments', np.mean),
                'mean_rev_score': ('review_score', np.mean),
                'mean_comment_length': ('review_comment_length', np.mean),
                'tot_comment_length': ('review_comment_length', np.sum),
                # 'cum_paytype': ('cum_paytype', most_freq_cat) ,
                # 'main_prod_categ': ('main_prod_categ', most_freq_cat),
                }
    # Dictionary for the aggreagation of dummy columns (payment and categories)
    cat_cols = list(df.columns[df.columns.str.contains('cat_')])
    pay_cols = list(df.columns[df.columns.str.contains('paytype_')])
    agg_dict_2 = {c+'_tot_nb': (c, np.sum) for c in cat_cols+pay_cols}

    # Concatenate the dictionaries
    agg_dict = dict(set(agg_dict_1.items()) | set(agg_dict_2.items()))

    # Aggregate the orders of each unique customer
    df_cust = df.groupby('customer_unique_id').agg(**agg_dict)

    df_cust['cust_region'] = df_cust['cust_region'].astype('object')

    # Changes the order of the columns
    cols = move_cat_containing(df_cust.columns, ['time', 'delay'], 'first')
    cols = move_cat_containing(cols, ['ord'], 'first')
    cols = move_cat_containing(cols, ['price', 'frei'], 'first')
    cols = move_cat_containing(cols, ['item'], 'first')
    cols = move_cat_containing(cols, ['cust'], 'first')
    cols = move_cat_containing(cols, ['pay'], 'last')
    cols = move_cat_containing(cols, ['pay_type'], 'last')
    cols = move_cat_containing(cols, ['cat_'], 'last')
    
    df_cust = df_cust[cols]
    return df_cust

''' Function that create the new features in the df_cust dataframe'''

def create_features_cust_df(df_cust):
    df_cust_mod = df_cust.copy('deep')
    # conversion of timedeltas
    time_cols = df_cust_mod.select_dtypes(include=['timedelta64[ns]']).columns
    for c in time_cols:
        df_cust_mod[c] = df_cust_mod[c].apply(lambda x: x.days)

    # Single purchase
    ser = (df_cust_mod['tot_nb_ord']==1).map({True: 1, False: 0})
    df_cust_mod.insert(loc=11, column='single_purch', value=ser)

    # Total number of not received orders
    ser = (df_cust_mod['tot_nb_ord']-df_cust_mod['tot_nb_deliv_ord'])
    df_cust_mod.insert(loc=12, column='nb_not_rec_orders', value=ser)

    # Average freight on payment value ratio
    ser = df_cust_mod['tot_freight_val']/(df_cust_mod['tot_pay_value']+1)
    df_cust_mod.insert(loc=8, column='avg_freight_payval_ratio', value=ser.fillna(0))
    return df_cust_mod

    '''calculates VIF and exclude colinear columns'''

from statsmodels.stats.outliers_influence import variance_inflation_factor    

def select_from_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if (max(vif) > thresh) and (len(variables)>2):
            print("nb of var:{}, max_vif={:.3} -> dropping \'{}\'"
                  .format(len(variables), max(vif),
                          str(X.iloc[:, variables].columns[maxloc])))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

'''Function that check low variance columns as a function of a variance threshold
 and returns a dataframe of results for easy plotting'''

from sklearn.feature_selection import VarianceThreshold

def check_feature_variance(thresholds, df_enc):
    n_cols = []
    old_excl_cols = []
    for th in thresholds:
        v_thresh = VarianceThreshold(threshold=th)
        df_filt = v_thresh.fit_transform(df_enc)
        mask_col = v_thresh.get_support(indices=True)
        excl_cols = [c for c in df_enc.columns if not c in df_enc.columns[mask_col]]
        new_excl_cols = [c for c in excl_cols if not c in old_excl_cols]
        old_excl_cols = excl_cols
        n_cols.append(df_filt.shape[1])
        print("thresh={} -> {} excluded cols and {} new ones: {}"\
                .format(th, len(excl_cols), len(new_excl_cols), new_excl_cols))
    df_res = pd.DataFrame({'thresh': thresholds,
                           'n_rem_cols': n_cols})
    return df_res

'''Computes the variable reduction of the data and plots the projected data on 1, 2 or 3 axis.
Choice of n_neibors, min_dist and metric available'''

from umap import UMAP 

def draw_umap(data, ser_clust=None, n_neighbors=15, min_dist=0.1, n_components=2,
              metric='euclidean', title='', fig=None, layout=None,
              figsize=(7,4), s=2, alpha=1, random_state=14):
    
    fit = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=1,
                    n_components=n_components, metric=metric,
                    random_state= random_state)
    u = fit.fit_transform(data)

    fig = plt.figure(figsize=figsize) if fig is None else fig
        
    colors = [sns.color_palette()[x] for x in ser_clust.astype('int')]\
        if ser_clust is not None else None

    layout=111 if layout is None else layout
        
    if n_components == 1:
        ax = fig.add_subplot(layout)
        ax.scatter(u[:,0], range(len(u)),
                   s=s,  alpha=alpha, c=colors)
    if n_components == 2:
        ax = fig.add_subplot(layout)
        ax.scatter(u[:,0], u[:,1],
                   s=s, alpha=alpha, c=colors)
    if n_components == 3:
        ax = fig.add_subplot(layout, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2],
                   s=s,  alpha=alpha, c=colors)

    plt.title(title, fontsize=12)

"""Checks if model object has any attributes ending with an underscore"""

import inspect

def is_fitted(model):
    return 0 < len( [k for k,v in inspect.getmembers(model) if k.endswith('_') \
                     and not k.startswith('__')] )


'''Computes the projection of the observations of df on the two first axes of
a transformation (PCA, UMAP or t-SNE)
The center option (clustering model needed) allows to project the centers
on the two axis for further display, and to return the fitted model
NB: if the model wa already fitted, does not refit.'''

from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE

def prepare_2D_axes(df, proj=['PCA', 'UMAP', 't-SNE'],
                    model=None, centers_on=False, random_state=14):

    dict_proj = dict()

    if centers_on:  # Compute and include the centers in the points
        model = model.fit(df) if not is_fitted(model) else model
        #### all clusterers don't have .cluster_centers method -> changed
        # centers = model.cluster_centers_ 
        # ind_centers = ["clust_" + str(i) for i in range(centers.shape[0])]
        # centers_df = pd.DataFrame(centers,
        #                           index=ind_centers,
        #                           columns=df.columns)
        #### all clusterers don't have .predict/labels_ method -> changed
        if hasattr(model, 'labels_'):
            clust = model.labels_
        else:
            clust = model.predict(df)
        centers_df = df.assign(clust=clust).groupby('clust').mean()

        df = df.append(centers_df)

    ## Projection of all the points through the transformations

    # PCA
    if 'PCA' in proj:
        pca = PCA(n_components=2, random_state=random_state)
        df_proj_PCA_2D = pd.DataFrame(pca.fit_transform(df),
                                      index=df.index,
                                      columns=['PC' + str(i) for i in range(2)])
        dict_proj = dict({'PCA': df_proj_PCA_2D})

    # UMAP
    if 'UMAP' in proj:
        umap = UMAP(n_components=2, random_state=random_state)
        df_proj_UMAP_2D = pd.DataFrame(umap.fit_transform(df),
                                       index=df.index,
                                       columns=['UMAP' + str(i) for i in range(2)])
        dict_proj = dict({'UMAP': df_proj_UMAP_2D})

    # t-SNE
    if 't-SNE' in proj:
        tsne = TSNE(n_components=2, random_state=random_state)
        df_proj_tSNE_2D = pd.DataFrame(tsne.fit_transform(df),
                                       index=df.index,
                                       columns=['t-SNE' + str(i) for i in range(2)])
        dict_proj = dict({'t-SNE': df_proj_tSNE_2D})

    # Separate the clusters centers from the other points if center option in on
    if centers_on:
        dict_proj_centers = {}
        for name, df_proj in dict_proj.items():
            dict_proj_centers[name] = dict_proj[name].loc[centers_df.index]
            dict_proj[name] = dict_proj[name].drop(index=centers_df.index)
        return dict_proj, dict_proj_centers, model
    else:
        return dict_proj


''' Plots the points on two axis (projection choice available : PCA, UMAP, t-SNE)
with clusters coloring if model available (grey if no model given).
NB: if the model wa already fitted, does not refit.'''


def plot_projection(df, model=None, proj='PCA', title=None,
                    figsize=(5, 3), palette='tab10', fig=None, ax=None, random_state=14):

    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    # if model : computes clusters, clusters centers and plot with colors
    if not model is None:

        # Computes the axes for projection with centers
        # (uses fitted model if already fitted)
        dict_proj, dict_proj_centers, model = prepare_2D_axes(df,
                                                              proj=[proj],
                                                              model=model,
                                                              centers_on=True,
                                                              random_state=random_state)
        # ...using model already fitted in prepare_2D_axes
        #### all clusterers don't have .predict/labels_ method -> changed
        if hasattr(model, 'labels_'):
            clust = model.labels_
        else:
            clust = model.predict(df)
        ser_clust = pd.Series(clust,
                              index=df.index,
                              name='Clust')

        n_clust = ser_clust.nunique()
        colors = sns.color_palette(palette, n_clust).as_hex()

        # Showing the points, cluster by cluster
        for i in range(n_clust):
            ind = ser_clust[ser_clust == i].index
            ax.scatter(dict_proj[proj].loc[ind].iloc[:, 0],
                       dict_proj[proj].loc[ind].iloc[:, 1],
                       s=1, alpha=0.7, c=colors[i])

            # Showing the clusters centers
            ax.scatter(dict_proj_centers[proj].iloc[:, 0].values[i],
                       dict_proj_centers[proj].iloc[:, 1].values[i],
                       marker='o', c=colors[i], alpha=0.4, s=150, edgecolor='k')
            # Showing the clusters centers labels
            ax.scatter(dict_proj_centers[proj].iloc[:, 0].values[i],
                       dict_proj_centers[proj].iloc[:, 1].values[i],
                       marker=r"$ {} $".format(str(i)),
                       c='k', alpha=1, s=70, )

    # if no model, only plot points in grey
    else:
        # Computes the axes for projection without centers
        dict_proj = prepare_2D_axes(df,
                                    proj=[proj],
                                    centers_on=False,
                                    random_state=random_state)
        # Plotting the point in grey
        ax.scatter(dict_proj[proj].iloc[:, 0],
                   dict_proj[proj].iloc[:, 1],
                   s=1, alpha=0.7, c='grey')

    title = "Projection: " + proj if title is None else title
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('ax 1'), ax.set_ylabel('ax 2')

"""Takes a clustering model (model), fits one time on the whole dataset (df)
('whole model'), and computes the labels for each row then for each
number of rows in a list (li_n_samp), refits the model and
computes the labels (.predict) for the whole dataset
Returns a dataframe (df_ARI_all_vs_sample_iter) containing all
the ARI scores obtained between the predictions made by the "whole model"
and that of each 'sample model' (columns of the dataframe)
NB: the same sequence is repeated a certain (n_iter) number of times
(rows of the dataframe)"""

from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

def check_ARI_through_sampling(model, df, li_n_samp,
                                  n_iter=10, stratify=None, print_opt=True):

    df_ARI_all_vs_sample_iter = pd.DataFrame()

    # Fit the model if not already done, compute clusters
    model = model.fit(df)
    ser_clust_all = pd.Series(model.labels_,
                          index=df.index,
                          name='all')

    # Looping over a number of iterations

    for i in range(n_iter):

        if print_opt: print(f"ooo ITERATION {i} ooo")

        # Looping over a list of sample indexes (with random_state changes)
        list_samp_df = []
        for n in li_n_samp:
            df_sampl, _ = \
                train_test_split(df, train_size=n, stratify=stratify,
                    random_state=14)
            list_samp_df.append(df_sampl)

        # Clustering labels obtained by fitting with samples
        # NB: the first column is the prediction of the "whole model"
        df_ARI_sampl = pd.DataFrame(ser_clust_all.to_frame())
        for df_samp in list_samp_df:
            n_samp = df_samp.shape[0]
            # print("ooooooooo Number of samples: ", n_samp)
            model.fit(df_samp)
            ser_clust = pd.Series(model.predict(df),
                                  index=df.index,
                                  name=str(n_samp)+'_sampl').to_frame()
            df_ARI_sampl = pd.concat([df_ARI_sampl, ser_clust], axis=1)

        # Computing the ARI score (whole dataset) between predictions with 
        # the "all" model vs predictions with "sample" model
        stab_sampl_kmeans = ARI_column_pairs(df_ARI_sampl, first_vs_others=True,
                                            print_opt=False)
        df_ARI_all_vs_sample_iter = pd.concat([df_ARI_all_vs_sample_iter,
                                               stab_sampl_kmeans.to_frame()],
                                              axis=1)
    df_ARI_all_vs_sample_iter.columns = \
            ['iter_'+str(i) for i in range(n_iter)]
    return df_ARI_all_vs_sample_iter


""" For a each number of clusters in a list ('list_n_clust'),
- runs iterations ('n_iter' times) of a KMeans on a given dataframe,
- computes the 4 scores : silhouette, davies-bouldin, calinski_harabasz and
distortion
- if enabled only('return_pop'): the proportion (pct) of the clusters
for each iteration and number of clusters
- and returns 3 dictionnaries:
    - dict_scores_iter: the 4 scores
    - dict_ser_clust_n_clust: the list of clusters labels for df rows
    - if enabled only (return_pop), 'dict_pop_perc_n_clust' : the proportions

NB: the functions 'plot_scores_vs_n_clust', 'plot_prop_clust_vs_nclust' and
'plot_clust_prop_pie_vs_nclust' plot
respectively:
- the scores vs the number of clusters,
- the proportion of clusters
- and the pies of the clusters ratio,
 based on the dictionnaries provided by compute_clust_scores_nclust"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def compute_clust_scores_nclust(df, list_n_clust=range(2,8),
                                 n_iter=10, return_pop=False):

    dict_pop_perc_n_clust = {}
    dict_ser_clust_n_clust = {}
    dict_scores_iter = {}

    # --- Looping on the number of clusters to compute the scores
    for i, n_clust in enumerate(list_n_clust,1):

        silh, dav_bould, cal_harab, distor = [], [], [], []
        pop_perc_iter, ser_clust_iter = pd.DataFrame(), pd.DataFrame()

        # Iterations of the same model (stability)
        for j in range(n_iter): 
            km = KMeans(n_clusters=n_clust, n_jobs=-1) # random_state not fixed !!!!
            km.fit(df)
            ser_clust = pd.Series(data=km.labels_,
                                  index=df.index, 
                                  name="iter_"+str(j))
            ser_clust_iter = pd.concat([ser_clust_iter, ser_clust.to_frame()],
                                       axis=1)

            if return_pop:
                # Compute pct of clients in each cluster
                pop_perc = 100 * ser_clust.value_counts() / df.shape[0]
                pop_perc.sort_index(inplace=True)
                pop_perc.index = ['clust_'+str(i) for i in pop_perc.index]
                pop_perc_iter = pd.concat([pop_perc_iter, pop_perc.to_frame()],
                                          axis=1)
        
            # Computing scores for iterations
            silh.append(silhouette_score(X=df, labels=ser_clust))
            dav_bould.append(davies_bouldin_score(X=df, labels=ser_clust))
            cal_harab.append(calinski_harabasz_score(X=df, labels=ser_clust))
            distor.append(km.inertia_)

        dict_ser_clust_n_clust[n_clust] = ser_clust_iter

        if return_pop:
            # dict of the population (pct) of clusters iterations
             dict_pop_perc_n_clust[n_clust] = pop_perc_iter.T

        # Dataframe of the results on iterations
        scores_iter = pd.DataFrame({'Silhouette': silh,
                                 'Davies_Bouldin': dav_bould,
                                 'Calinsky_Harabasz': cal_harab,
                                 'Distortion': distor})
        dict_scores_iter[n_clust] = scores_iter

    if return_pop:
        return dict_scores_iter, dict_ser_clust_n_clust, dict_pop_perc_n_clust
    else:
        return dict_scores_iter, dict_ser_clust_n_clust


''' Plot the 4 mean scores stored in the dictionnary returned by the function
compute_clust_scores_nclust (dictionnary of dataframes of scores (columns)
for each iteration (rows) of the model and for each number of clusters
in a figure with error bars (2 sigmas)'''

def plot_scores_vs_n_clust(dict_scores_iter, figsize=(15,3)):

    fig = plt.figure(figsize=figsize)
    list_n_clust = list(dict_scores_iter.keys())

    # Generic fonction to unpack dictionary and plot one graph
    def score_plot_vs_nb_clust(scores_iter, name, ax, c=None):
        score_mean = [dict_scores_iter[i].mean().loc[n_score] for i in list_n_clust]
        score_std = np.array([dict_scores_iter[i].std().loc[n_score]\
                            for i in list_n_clust])
        ax.errorbar(list_n_clust, score_mean, yerr=2*score_std, elinewidth=1,
                capsize=2, capthick=1, ecolor='k', fmt='-o', c=c, ms=5,
                barsabove=False, uplims=False)

    li_scores = ['Silhouette', 'Davies_Bouldin',
                 'Calinsky_Harabasz', 'Distortion']
    li_colors = ['r', 'b', 'purple', 'g']

    # Looping on the 4 scores
    i=0
    for n_score, c in zip(li_scores, li_colors):
        i+=1
        ax = fig.add_subplot(1,4,i)
        
        score_plot_vs_nb_clust(dict_scores_iter, n_score, ax, c=c)
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel(n_score+' score')

    fig.suptitle('Clustering score vs. number of clusters',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


''' Plot the proportion (%) of each cluster (columns) returned by the function
compute_clust_scores_nclust (dictionnary of dataframes of the proportion
for each iteration (rows) of the model in one figure with error bars (2 sigmas)'''

def plot_prop_clust_vs_nclust(dict_pop_perc_n_clust, figsize=(15,3)):

    fig = plt.figure(figsize=figsize)
    list_n_clust = list(dict_pop_perc_n_clust.keys())
    

    for i, n_clust in enumerate(list_n_clust, 1):
        n_iter = dict_pop_perc_n_clust[n_clust].shape[0]
        ax = fig.add_subplot(3,3,i)
        sns.stripplot(data=dict_pop_perc_n_clust[n_clust],
                      edgecolor='k', linewidth=1,  ax=ax)
        ax.set(ylim=(0,100))
        ax.set_ylabel("prop. of the clusters (%)")
    fig.suptitle(f"Proportion of the clusters through {n_iter} iterations",
                fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.97])


""" Plot pies of the proportion of the clusters using the proportions
stored in the dictionnary returned by the function
'compute_clust_scores_nclust' (dictionnary of dataframes of the
proportions (columns) for each iteration (rows) of the model
and for each number of clusters in a figure with error (+/-2 sigmas)"""

def plot_clust_prop_pie_vs_nclust(dict_pop_perc_n_clust,
                                  list_n_clust, figsize=(15, 3)):

    fig = plt.figure(figsize=figsize)

    for i, n_clust in enumerate(list_n_clust,1):
        ax = fig.add_subplot(str(1) + str(len(list_n_clust)) + str(i))

        mean_ = dict_pop_perc_n_clust[n_clust].mean()
        std_ = dict_pop_perc_n_clust[n_clust].std()
        
        wedges, texts, autotexts = ax.pie(mean_,
                autopct='%1.0f%%',
                labels=["(+/-{:.0f})".format(i) for i in std_.values],
                pctdistance=0.5)
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=8)
        ax.set_title(f'{str(n_clust)} clusters')  # , pad=20

    fig.suptitle('Clusters ratio', fontsize=16, fontweight='bold')
    plt.show()


''' Plots on the left the silhouette scores of each cluster and
on the right the projection of the points with cluster labels as cluster'''

from sklearn.metrics import silhouette_score, silhouette_samples

def silh_scores_vs_n_clust(df, n_clust, proj='PCA',
                           xlim=(-0.1, 0.8), figsize=(18, 3), palette='tab10'):
    
    palette = sns.color_palette(palette, np.max(n_clust))
    colors = palette.as_hex()

    distor = []
    for n in n_clust:
        fig = plt.figure(1, figsize=figsize)

        # --- Plot 1: Silhouette scores
        ax1 = fig.add_subplot(121)

        model = KMeans(n_clusters=n, random_state=14)
        model = model.fit(df)

        ser_clust = pd.Series(model.predict(df),
                              index=df.index,
                              name='Clust')
        distor.append(model.inertia_)
        sample_silh_val = silhouette_samples(df, ser_clust)

        y_lower = 10
        # colors = [colors[x] for x in ser_clust.astype('int')]
        for i in range(n):
            # Aggregate and sort silh scores for samples of clust i
            clust_silh_val = sample_silh_val[ser_clust == i]
            clust_silh_val.sort()
            size_clust = clust_silh_val.shape[0]
            y_upper = y_lower + size_clust
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0,
                              clust_silh_val,
                              facecolor=colors[i],
                              edgecolor=colors[i],
                              alpha=0.7)

            # Label of silhouette plots with their clust. nb. at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_clust, str(i))

            # Computes the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        silhouette_avg = silhouette_score(df, ser_clust)
        ax1.set_title("Nb of clusters: {} | Avg silhouette: {:.3f}" \
                      .format(n, silhouette_avg), fontsize=12)
        ax1.set_xlabel("Silhouette coeff. values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_xlim(list(xlim))
        # (n+1)*10: inserting blank spaces between clusters silh scores
        ax1.set_ylim([0, df.shape[0] + (n + 1) * 10])

        # --- Plot 2: Showing clusters on chosen projection
        ax2 = fig.add_subplot(122)
        # uses already fitted model
        plot_projection(df, model=model,
                        proj=proj,
                        palette=palette,
                        fig=fig, ax=ax2)

        ax2.set_title('projection: ' + proj, fontsize=12)

        plt.suptitle("Silhouette analysis for {} clusters".format(n),
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


''' Generates the list of all unique combination of k numbers 
(no matter the order) among a given seq list of objects'''

def combinlist(seq, k):
    p = []
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                s.append(seq[j])
            j += 1
        if len(s)==k:
            p.append(s)
        i += 1
    return p


'''Takes a dataframe of clusters number (prediction) for a set of observation, 
and computes the ARI score between pairs of columns.
Two modes are available:
- first_vs_others=False: to check the initialisation stability.
The columns are obtains for n_columns iterations of the same model
with different initialisation
- first_vs_others=True: to compare the predictions obtained with the whole
dataset (first column) and predictions obtained with a sample
(the other columns)
Return a pd.Series of the ARI scores (values) for each pair of columns (index).
'''

from sklearn.metrics import adjusted_rand_score

def ARI_column_pairs(df_mult_ser_clust, first_vs_others=False, print_opt=True):

    n_columns = len(df_mult_ser_clust.columns)
    n_clust = df_mult_ser_clust.stack().nunique()
    
    # Computes ARI scores for each pair of models
    ARI_scores = []
    if first_vs_others: # first columns versus the others
        pairs_list = [[df_mult_ser_clust.columns[0],
                       df_mult_ser_clust.columns[i]] \
                      for i in range(1, n_columns)]
        if print_opt: print("--- ARI between first and the {} others ---"\
                            .format(n_columns-1))
        name = f'ARI_{str(n_clust)}_clust_first_vs_others'
    else: # all pairs
        pairs_list = combinlist(df_mult_ser_clust.columns,2)
        if print_opt: print("--- ARI all {} unique pairs ---"\
                            .format(len(pairs_list)))
        name = f'ARI_{str(n_clust)}_clust_all_pairs'

    for i, j in pairs_list:
        ARI_scores.append(adjusted_rand_score(df_mult_ser_clust.loc[:,i],
                                              df_mult_ser_clust.loc[:,j]))

    # Compute the mean and standard deviation of ARI scores
    ARI_mean, ARI_std = np.mean(ARI_scores), np.std(ARI_scores)
    ARI_min, ARI_max = np.min(ARI_scores), np.max(ARI_scores)
    if print_opt: print("ARI: mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f} "\
            .format(ARI_mean, ARI_std, ARI_min, ARI_max))

    return pd.Series(ARI_scores, index=pd.Index(pairs_list),
                     name=name)


''' For each quantitative value of the original dataframe
(prior to transformation and clustering), returns two dataframes:
- the mean value for each clusters
- the mean of the whole values
- the relative difference of the means between clusters and whole dataframe '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.cluster import KMeans

def mean_deviation_clust(model, df, orig_df, palette='seismic', figsize=(20, 3)):

    # Filters the numeric datas in 'orig_df'
    orig_df_quant = orig_df.select_dtypes(include=[np.number])
    model = model.fit(df) if not is_fitted(model) else model    

    # Assign segment to each customer in original dataset
    data_with_clust = orig_df_quant.assign(cluster=model.labels_)
    k = data_with_clust['cluster'].nunique()

    # Compute average for each feature by cluster
    clust_mean = data_with_clust.groupby(['cluster']).mean().round(2)

    # Ratio of difference between mean and cluster means for each feature
    orig_df_mean = orig_df_quant.mean()
    rel_var = 100 * (clust_mean - orig_df_mean) \
                    / (orig_df_mean + 0.1)

    # Plotting figure
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    vlim = np.array([abs(rel_var.min().min()),
                     abs(rel_var.max().max())]).max()
    sns.heatmap(data=rel_var,
                vmin=-vlim, vmax=vlim,
                center=0, annot=True, fmt='.0f',
                cmap=palette, ax=ax1)
    ax1.set_title('Mean deviation to the mean (%)',
        fontsize=14, fontweight='bold' ,pad=20)
    ax1.set_ylabel(ylabel='cluster', labelpad=20)

    return clust_mean, orig_df_mean, rel_var


''' Plotting ANOVA and Kruskall-Wallis H test and return if same dist. or not
i.e., at least one of the groups has a significantly different mean from the others.'''

from scipy.stats import f_oneway, kruskal

def test_distrib_clust(data_df, Q_col, C_col, print_opt=True):
    gb = data_df.groupby(C_col)[Q_col]
    cat = list(gb.groups.keys())
    cat_series = [gr.dropna().values for n, gr in gb]

    if print_opt: print(5 * 'ooo' + ('- {} vs. {} -'.format(Q_col, C_col)) + 5 * 'ooo')

    # Analysis of Variance Test
    stat1, p1 = f_oneway(*cat_series)
    if print_opt:
        print('---ANOVA---')
        print('stat=%.3f, p=%.10f' % (stat1, p1))
        print('Prob. same distr') if p1 > 0.05 else print('Prob. different distr')
    # Kruskal-Wallis H Test
    stat2, p2 = kruskal(*cat_series)
    if print_opt:
        print('---Kruskal-Wallis---')
        print('stat={:.3f}, p={:.10f}'.format(stat2, p2))
        print('Prob. same distr') if p2 > 0.05 else print('Prob. different distr')

    return pd.Series({'ANOVA': (p1, str(p1 > 0.05)), # round(p1, 4)
            'Kruskal-Wallis': (p2, str(p2 > 0.05))},
            name=Q_col) # round(p2, 4)


'''Summary of results of ANOVA and Kruskal Wallis test on each
quantitative column of data_df against one categorical columns (C_col)
''' 

def summary_ANOVA_Kruskal(data_df, C_col, print_opt=True):

    res_df=pd.DataFrame()
    for c in data_df.select_dtypes(include=np.number).columns:
        if print_opt: print('oooo--'+c+'--oooo')
        dic_stat = test_distrib_clust(data_df, C_col=C_col,
                                            Q_col=c, print_opt=print_opt)
        for s in dic_stat.keys():
            res_df.loc[s,c] = str(dic_stat.get(s))
    return res_df

''' Computes the clusters from a model (does not refit if already fitted)
and creates contingency tables of binarized quantitative data vs. clusters
- df is the transformed dataset on which the clustering is or will be fitted
- df_expl is the same dataset, prior to transformation'''

from scipy.stats import chi2_contingency


def contingency_tables(model, df, df_expl, min_max=None,
                       cut_mode='uniform', # 'uniform' or 'quantile'
                       palette="seismic"):

    # Select only the quantitative data prior to transformation
    df_expl_quant = df_expl.select_dtypes(include=[np.number])

    # Nb of graphs, and rows of graphs
    nb_ax = len(df_expl_quant.columns)
    n_rows = (nb_ax+1)//2
    fig = plt.figure(figsize=(12,3*n_rows))

    # Fit the model if not already done, compute clusters
    model = model.fit(df) if not is_fitted(model) else model
    ser_clust = pd.Series(model.labels_,
                          index=df.index,
                          name='Clust')

    for i, col in enumerate(df_expl_quant.columns, 1):
        # Binarize the quantitative data
        if cut_mode == 'quantile':
            ser_bin = pd.qcut(df_expl_quant[col], [0,0.2,0.4,0.6,0.8,1],
                              precision=2, duplicates='drop')
        elif cut_mode == 'uniform':
            ser_bin = pd.cut(df_expl_quant[col], bins=5, precision=2)
        else:
            print("ERROR: cut_mode unknown, use 'quantile' or 'uniform'")
        data_crosstab = pd.crosstab(ser_clust, ser_bin, margins = False)

        ## Compute and print Chi-sqare score
        stat, p, dof, expected = chi2_contingency(data_crosstab)
        res_str = 'Probably indep.' if p > 0.05 else 'Probably dep.'
        res_str = 'Chi²: stat={:.3f}, p={:.3f}, {}'.format(stat, p, res_str)

        # Plot a grid of tables of contingency
        ax = fig.add_subplot(n_rows, 2, i)
        if min_max == None:
            vmin = data_crosstab.min().min()
            vmax = data_crosstab.max().max()
        else:
            vmin, vmax = min_max
        center = (vmin + vmax) / 2
        plot_heatmap(data_crosstab, vmin=vmin, center=center, vmax=vmax,
                     title=col, palette=sns.color_palette(palette, 20),
                     shape='rect', fmt='.0f', fig=fig, ax=ax)
        
        # if "precision" does not work: truncate the floats in the labels
        digit_round = 2
        list_interval = [x.get_text() for x in ax.get_xticklabels()] 
        str_intervals = [i.replace("(","").replace("]", "").split(", ")\
                            for i in list_interval]
        rounded_cuts = ["["+str(round(float(i),digit_round))\
                            +", "+str(round(float(j),digit_round))+")"\
                            for i, j in str_intervals]#[:-1]]+['All']
        ax.set_xticklabels(rounded_cuts)

    plt.tight_layout()
    plt.show()


    ''' Plots a radar chart of the cluster profiles from a dataframe containing:
- the name or number of cluster as index
- the means of each features per clusters in columns
OR
- the value of each features (transformed) for each cluster center (centroid)
NB: the values would be the same if the was no transformation

Values are scaled using a MinMaxScaler'''

from sklearn.preprocessing import MinMaxScaler
from math import pi

def plot_radar_chart(df, row, title, color, min_max_scaling=False, ax=None):

    df_copy = df.copy('deep')
    if min_max_scaling:
        min_max = MinMaxScaler()
        df_copy = pd.DataFrame(min_max.fit_transform(df_copy),
                               index=df_copy.index,
                               columns=df_copy.columns)
    df_ = df_copy.reset_index()
    categories=list(df)
    n_vars = len(categories)
    
    angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
    angles += angles[:1] # "complete the loop"
    
    ax = plt.subplot(1,1,1, polar=True) if ax is None else ax
    
    # First axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=7)# [10,20,30], ["10","20","30"], 
    # plt.ylim(0,40)
    
    values=df_.loc[row].drop('clust').values.flatten().tolist() # 
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
    
    plt.title(title, size=11, color=color, y=1.1,
              fontweight='bold')


''' Class that join a clustering algorithm to a classification algorithm
to be able to train the clustering algorithm on a sample and to predict
clusters labels on new unlearned data '''

from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.metaestimators import if_delegate_has_method

class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X):
        return self.classifier_.predict(X)

    @if_delegate_has_method(delegate='classifier_')
    def decision_function(self, X):
        return self.classifier_.decision_function(X)


''' Plots boxplots of quantitative features for each cluster'''

def plot_boxplots_feat_vs_clust(df, df_expl, model, col_order=None):

    fig = plt.figure(figsize=(12,12))

    model = model.fit(df) if not is_fitted(model) else model
    ser_clust = pd.Series(model.labels_,
                        index=df.index,
                        name='clust')

    with sns.color_palette('dark'):
        col_order = df.columns if col_order is None else col_order
        for i, c in enumerate(col_order,1):
            ax = fig.add_subplot(4,4,i)
            sns.boxplot(data=df_expl.assign(clust=ser_clust),
                        x='clust', y=c, width=0.5, ax=ax)
            plt.grid()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Dispersion of quantitative data through clusters',
                fontsize=16, fontweight='bold')
    plt.show()

''' Plots the Snake plot from a dataframe ('rel_var') of the relative
deviation of the mean (pct) of the features (columns) for each cluster (index)
NB: 'rel_var' can be obtained using the 'mean_deviation_clust' function
('rel_var' output)
The most important columns can be selected using 'thresh_dev'.
'''

from sklearn.preprocessing import StandardScaler

def plot_snake(rel_var, thresh_dev=5):

    sel_col = rel_var[rel_var>thresh_dev].dropna(how='all', axis=1).columns
    sel_col = [col for col in sel_col if not 'cat_' in col]
    n_clust = rel_var.shape[0]

    ss = StandardScaler()
    std_rel_var = pd.DataFrame(ss.fit_transform(rel_var.T).T,
                               index = rel_var.index,
                               columns=rel_var.columns)

    fig, ax = plt.subplots(1)
    std_rel_var[sel_col].plot(marker='o', ax=ax,
                              color=sns.color_palette('tab10'))
    ax.legend(ncol=1, bbox_to_anchor=[1,1])
    plt.xticks(np.arange(n_clust))
    ax.set_ylabel("standardized rel. mean dev.")
    plt.title("Snake plot", fontweight='bold')
    fig.set_size_inches(6,4)
    plt.show()

''' SANKEY DIAGRAM | Takes the dataframe of the clusters labels of the customers for 
each period of time, then computes the flows between the clusters of each period
returns a dataframe of the source label, target label and
value of the flow (to be used with go.Sankey)
'''
def computing_flows_dataframe(df_clust_1st_year):

    n_clust = df_clust_1st_year.nunique().max()
    n_periods = df_clust_1st_year.shape[1]
    # Renaming the columns with period number for easier handling
    df_ = df_clust_1st_year
    df_.columns = np.arange(len(df_.columns))

    all_flows = pd.DataFrame()
    for i in range(n_periods-1):
        # Computing the flow for each clusters pair from i period to the next
        flow = df_.groupby([i,i+1]).size()
        flow = flow.rename(f'flow_{i}_{i+1}').to_frame()
        # Adding lines where there is no flow, setting to 0
        # (useful later for color attribution)
        new_index = [(a,b) for a in range(n_clust) for b in range(5)]
        flow = flow.reindex(new_index).fillna(0).reset_index()
        flow = flow.rename(columns={i:'source', i+1:'target'})
        # Adding to 'source' and 'target' an offset in order to correspond to
        # the labels of the Sankey diagram nodes
        flow['source'] = flow['source'].add(n_clust*i)
        flow['target'] = flow['target'].add(n_clust*(i+1))
        flow.columns = ['source', 'target', 'flows']
        all_flows = pd.concat([all_flows, flow], ignore_index=True)

    return all_flows

''' SANKEY DIAGRAM | From the dataframe of the clusters labels of the customers for 
each period of time, computes flow data and plots the Sankey diagram.
'''

import plotly.graph_objects as go

def plot_Sankey_diagram(df_clust_1st_year, title):

    # Computes the flows between the clusters of each period
    all_flows = computing_flows_dataframe(df_clust_1st_year)

    # Looping over periods and clusters labels to create Sankey labels
    n_clust = df_clust_1st_year.nunique().max()
    n_periods = df_clust_1st_year.shape[1]
    labels = []
    for i in range(n_periods):
        for j in range(n_clust):
            labels.append(f'c_{j}_p_{i}')

    # Defining the list of colors for the nodes and for the flows
    colors_nodes = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{1})'\
            for r,g,b in sns.color_palette('tab10')[:n_clust]]*n_clust*n_periods
    # colors_flows = [[f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{0.4})']*n_clust\
    #         for r,g,b in sns.color_palette('tab10')[:n_clust]]*n_periods
    # colors_flows = [col for subl in colors_flows for col in subl]

    # Plotting the Sankey diagram
    fig = go.Figure(data=[go.Sankey(valueformat = ".0f",
                                    valuesuffix = "customers",
                                    node = dict(pad = 10,
                                                thickness = 20,
                                                line = dict(color = "grey",
                                                            width = 0.5),
                                                label = labels,
                                                color = colors_nodes),
                                    link = dict(source = all_flows['source'],
                                                target = all_flows['target'],
                                                value = all_flows['flows'],
                                                color = 'rgba(10,10,10,0.2)',
                                                ))])

    fig.update_layout(title_text=title, font_size=10)
    fig.show()

    return all_flows


    ''' Takes the dictionnary of all the transformed dataframes of all the periods,
then runs all the step of the stability analysis (KMeans) of the
clustering through time:
- Computes the cluster label of each customer of the 1st year for each period
- Computes ARI (clusters obtained with first period vs other periods)
- Computes flows between clusters
- Plots Sankey diagram
NB: two modes are available
method1: uses one model fitted on the data of the first year only
method2: refits the model on each new database
'''

def stability_through_time(dict_df_trans, n_clust, method1=True, method2=True):

    # Indexes of the customers of the first year
    ind_1st_year = dict_df_trans[0][0].index

    if method1:
        print("Method 1: Following customers of the first year with the \
same model (using .predict)")
        # Fitting the model on the database of the first year only
        model_M1 = KMeans(n_clusters=n_clust, random_state=14)
        model_M1.fit(dict_df_trans[0][0])
        # Computing cluster label of each customer of 1st year for each period
        df_clust_1st_year_M1 = pd.DataFrame()
        for k, v in dict_df_trans.items():
            df_ = v[0] # database of period k
            name_ = v[1] # name of period k
            ser_clust = pd.Series(model_M1.predict(df_),
                                  index=df_.index,
                                  name=f'{k}_clust_'+str(name_))
            ser = ser_clust.loc[ind_1st_year]
            df_clust_1st_year_M1 = pd.concat([df_clust_1st_year_M1,
                                              ser.to_frame()],
                                        axis=1)
        # Computing the ARI between the clusters obtained with first period and
        # each of the other periods (fitted once only)
        ser_ARI_1_month_M1 = \
            ARI_column_pairs(df_clust_1st_year_M1,
                            first_vs_others=True, print_opt=False)
        # Plotting ARIs
        fig = plt.figure(figsize=(5,3))
        ax = fig.add_subplot(111)
        ax.plot(range(1, 14), ser_ARI_1_month_M1.values, '-or')
        ax.set_xlabel('nb of extra months added')
        ax.set_ylabel('ARI (1st year as ref)')
        plt.title('Comparison of clusters labels (model fitted once)',
                  fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
        # Computing flow data and plotting the Sankey diagram
        title="Sankey Diagram showing flows of customers from one period\
 to another - (model fitted once)"
        all_flows = plot_Sankey_diagram(df_clust_1st_year_M1, title=title)

    if method2:
        print("Method 2: Following customers of the first year with the \
model refitted for each period")
        # Computing the cluster label of each customer for each period
        # first initialisation
        init_kmeans = 'k-means++'
        df_clust_1st_year_M2 = pd.DataFrame()
        for k, v in dict_df_trans.items():
            df_ = v[0] # database of period k
            name_ = v[1] # name of period k
            n_init = 10 if k==0 else 1 # to avoid warning
            # init of a new model with init using centroids of the last period
            model_M2 = KMeans(n_clusters=n_clust, init=init_kmeans,
                              n_init=n_init, random_state=14)
            # refitting the model for each period of time (with new customers)
            model_M2.fit(df_)
            # computing cluster_centers for more accurate fit on next period
            init_kmeans = model_M2.cluster_centers_
            # getting labels of all the customers of the period
            ser_clust = pd.Series(model_M2.predict(df_),
                                  index=df_.index,
                                  name=f'{k}_clust_'+str(name_))
            # storing only labels of the customers of 1st year
            ser = ser_clust.loc[ind_1st_year]
            df_clust_1st_year_M2 = pd.concat([df_clust_1st_year_M2,
                                              ser.to_frame()],
                                        axis=1)
        # Computing the ARI between the clusters obtained with first period and
        # each of the other periods (refit at each time)
        ser_ARI_1_month_M2 = ARI_column_pairs(df_clust_1st_year_M2,
                                first_vs_others=True, print_opt=False)
        # Plotting ARIs
        fig = plt.figure(figsize=(5,3))
        ax = fig.add_subplot(111)
        ax.plot(range(1, 14), ser_ARI_1_month_M2.values, '-ob')
        ax.set_xlabel('nb of extra months added')
        ax.set_ylabel('ARI (1st year as ref)')
        plt.title('Comparison of clusters labels (model re-fitted each time)',
                  fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
        # Computing flow data and plotting the Sankey diagram
        title="Sankey Diagram showing flows of customers from one period \
to another - (model re-fitted each time)"
        all_flows = plot_Sankey_diagram(df_clust_1st_year_M2, title=title)

''' Takes a dataset (df_expl) and the same dataset transformed prior
to clustering.
Then runs:
- visualisation of the distribution of the features
- evaluation of sampling relevancy
- optimisation of the number of cluster (10 iterations)
    o showing 4 clustering scores
    o showing population ratio
    o silhouette of each clusters
    o initialisation stability ARI (20 iterations)
-> asking the user for best number based on scores, population of clusters
and initialisation stability
-  cluster analysis
    o visualization of the clusters on PCA, UMAP, t-SNE projections
    o contingency tables
    o relative difference
'''


def kmeans_clustering_all_steps(df, df_expl, stratify=None):
    
    ################## SHOWING CLUSTERING DATASET ##################
    # Histograms of the untransformed data
    print('ooooooooooooo- UNTRANSFORMED DATA (df_expl) -ooooooooooooo')
    plot_histograms(df=df_expl, cols=df_expl.columns,
                    figsize=(12,15), bins=30, layout=(9,4))
    # Histograms of the transformed data
    print('ooooooooooooo- TRANSFORMED DATA (df) -ooooooooooooo')
    plot_histograms(df=df, cols=df.columns,
                    figsize=(12,15), bins=30, color='pink', layout=(9,4))

    ######################## SAMPLING RELEVANCY ########################
    # Choosing a model for sampling relevancy
    km_sampl = KMeans(n_clusters=6, random_state=14)
    km_sampl.fit(df)
    # Checking the ARI score between predictions of the "whole model"
    # vs. the "sample model", n_iter times.
    n_iter=10
    li_n_samp = [100, 250, 500, 750, 1000, 2000, 3000,
                4000, 5000, 7500, 10000, 20000, 50000]
    # Bining the mean review score column for further stratification (sampling)
    if stratify is None:
        stratify = pd.cut(df_expl['mean_rev_score'], [0,1,2,3,4,5])
    df_ARI_all_vs_sample_iter = \
        check_ARI_through_sampling(km_sampl, df, li_n_samp, n_iter=n_iter,
                                   stratify=stratify, print_opt=False)
    # Plotting the results
    fig, ax = plt.subplots(1)
    fig.set_size_inches(13,5)
    ax.errorbar(li_n_samp,
                df_ARI_all_vs_sample_iter.mean(1).values,
                yerr=2*(df_ARI_all_vs_sample_iter.std(1).values),
                marker='o', color='blue')
    ax.tick_params(rotation=45)
    ax.set_xscale('log')
    plt.xlabel("Number of rows in the sample")
    plt.ylabel("ARI score")
    plt.title(f"Pred. of model fitted with the whole dataset \
    vs. pred. of model fitted with a sample ({n_iter}) iterations",
    fontweight='bold')
    plt.grid()
    plt.show()

    # Asking the user to enter the sample size
    sampl_size = int(input("Please, choose a convenient sample size: "))
    df_sampl, _ = train_test_split(df, train_size=sampl_size,
                                   stratify=stratify, random_state=14)
    ind_sampl = df_sampl.index

    ############## OPTIMISATION OF THE NUMBER OF CLUSTERS ##############
    # Reducing the size of the dataset
    df = df.loc[ind_sampl]
    df_expl = df_expl.loc[ind_sampl]
    # Choosing the number of clusters range
    list_n_clust = range(2,9)
    # Computes and returns:
    # - the aggregated results (mean, median, std) of the 4 scores
    # - the list of the clusters predicted for each iter. and nb of clusters
    # - the proportion (pct) of the clusters 
    dict_pkl_A = {}
    n_iter = 20
    dict_scores_iter, dict_ser_clust_n_clust, dict_pop_perc_n_clust = \
                        compute_clust_scores_nclust(df,
                                                    list_n_clust=list_n_clust,
                                                    n_iter=n_iter,
                                                    return_pop=True)
    # Plotting the 4 scores results              
    plot_scores_vs_n_clust(dict_scores_iter, figsize=(15,3))
    # Plotting the proportion of clusters (pies)
    plot_clust_prop_pie_vs_nclust(dict_pop_perc_n_clust,
                                list_n_clust, figsize=(15, 3))
    # Computing and plotting the silhouette score of each cluster
    silh_scores_vs_n_clust(df, n_clust=list_n_clust, proj='t-SNE',
                        xlim=(-0.1,1), figsize=(8,4),
                        palette='tab10')
    # Checking for initialisation stability of the clusters
    df_ARI_stab = pd.DataFrame()
    for i in list_n_clust:
        stab_init_kmeans = ARI_column_pairs(dict_ser_clust_n_clust[i],
                                            first_vs_others=False,
                                            print_opt=False)
        df_ARI_stab = pd.concat([df_ARI_stab, stab_init_kmeans.to_frame()],
                                axis=1)
    # Boxplot of the ARI score on multiples iterations
    df_ARI_stab.boxplot(color='red', vert=False)
    plt.gcf().set_size_inches(7,2.5)
    plt.title('Initialisation stability', fontweight='bold')
    plt.show()

    # Asking the user to enter the clusters number for the best model
    n_clust = int(input("Please, choose the number of\
     clusters for the best model: "))
    
    ########################### CLUSTERS ANALYSIS ###########################
    # Visualisation of the best clusters
    # fit the model on the whole dataframe
    best_model = KMeans(n_clusters=n_clust, random_state=14)
    best_model.fit(df)
    ser_clust = pd.Series(data=best_model.labels_,
                        index=df.index)
    silh = silhouette_score(X=df, labels=ser_clust)
    dav_bould = davies_bouldin_score(X=df, labels=ser_clust)
    cal_harab = calinski_harabasz_score(X=df, labels=ser_clust)
    distor = best_model.inertia_
    print(f"scores of the best model : \n silh={silh}, \
    cal_har={cal_harab}, dav_bould={dav_bould}, distor={distor}")
    # ----- plotting the clusters on PCA, UMAP and t-SNE projections -----
    fig = plt.figure(figsize=(12,3))
    tab_proj = ['PCA', 'UMAP', 't-SNE']
    for i, proj in enumerate(tab_proj,1):
        ax = fig.add_subplot(1,len(tab_proj), i)
        # plot only a sample, but using the model already fitted
        plot_projection(df, model=best_model, proj=proj,
                        fig=fig, ax=ax)
    fig.suptitle("Projection of the clusters of the best model",
                fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.92])
    plt.show()
    ## ----- Contingency tables -----
    best_model.fit(df)
    ser_clust = pd.Series(best_model.labels_,
                            index=df.index,
                            name='clust')
    df_expl_quant = df_expl.select_dtypes(include=[np.number])
    ser_bin = pd.qcut(df_expl_quant[df_expl_quant.columns[0]],
                      [0,0.2,0.4,0.6,0.8,1],
                      precision=2, duplicates='drop')
    data_crosstab = pd.crosstab(ser_clust, ser_bin, margins = False)
    contingency_tables(best_model, df, df_expl, palette="mako",
                    cut_mode='quantile')
    ## ----- Relative difference to the mean -----
    clust_mean, orig_df_mean, rel_var = \
        mean_deviation_clust(best_model, df, df_expl.loc[df.index],
                             palette='seismic', figsize=(15,2))
    ## ----- Snake plot -----
    plot_snake(rel_var, thresh_dev=5)
    ## ----- ANOVA and Kruskal-Wallis -----
    # Computing ANOVA and Kruskal-Wallis for each features against clusters
    Anova_Kruskal_df = pd.DataFrame()
    for i, col in enumerate(df_expl.columns,1):
        ser = test_distrib_clust(df_expl.assign(clust=ser_clust), col,
                        'clust', print_opt=False)
        Anova_Kruskal_df = pd.concat([Anova_Kruskal_df, ser.to_frame()], axis=1)
    # Plotting the results
    fig, ax = plt.subplots(1)
    ser_anova = pd.Series([a for a,b in Anova_Kruskal_df.loc['ANOVA'].values],
                          index=Anova_Kruskal_df.loc['ANOVA'].index,
                          name='anova')
    ser_kruskal = pd.Series([a for a,b in \
                                Anova_Kruskal_df.loc['Kruskal-Wallis'].values],
                            index=Anova_Kruskal_df.loc['Kruskal-Wallis'].index,
                            name='kruskal')
    df_ = pd.concat([ser_anova, ser_kruskal], axis=1).sort_values('anova')

    (df_['anova'].sort_values()+1e-295).plot(marker='o', color='red',
                                             label='ANOVA', ax=ax)
    (df_['kruskal'].sort_values()+1e-295).plot(marker='o', color='blue',
                                               label='Kruskal-Wallis', ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels(df_.index)
    plt.xticks(rotation=45, ha='right')
    plt.xticks(np.arange(ser_anova.shape[0]))
    plt.title('p-value of the stat.hypothesis test\n (independence)',
            fontweight='bold', fontsize='14')
    plt.legend()
    plt.show()
    # Plotting boxplot of quantitative features for each cluster
    # NB: col_order -> to keep the order of the lower p-value first
    plot_boxplots_feat_vs_clust(df, df_expl, best_model, col_order=df_.index)
    ## ----- Radar charts -----
    ser_clust = best_model.labels_
    # DataFrame with the means of each columns for each cluster
    df_clust = df_expl.assign(clust=ser_clust)\
        .reindex(columns=['clust']+list(df_expl.columns))
    df_clust_mean = df_clust.groupby('clust').mean()
    # Plotting the radar chart (untransformed data)
    my_dpi = 96
    n_row = (n_clust+1)//2
    fig = plt.figure(figsize=(n_row*350/my_dpi, 500/my_dpi),
                    dpi=my_dpi)
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Dark2", n_clust)
    # Loop to plot
    for i, row in enumerate(df_clust_mean.index, 1):
        ax = fig.add_subplot(2,n_row,i, polar=True)
        plot_radar_chart(df=df_clust_mean, row=row,
                        title='cluster '+str(row),
                        color=my_palette(row),
                        min_max_scaling=True, ax=ax)
    plt.tight_layout(rect=[0,0,1,0.95])
    ## ----- Decision Tree -----
    thresh_dev = 0
    sel_col = rel_var[np.abs(rel_var)>thresh_dev].dropna(how='all', axis=1).columns
    sel_col = [col for col in sel_col if not 'cat_' in col]
    df_dec_tree = df_clust[sel_col+['clust']]
    X_tr, y_tr = df_dec_tree.iloc[:,:-1], df_dec_tree.iloc[:,-1]
    # initializing and fitting the tree
    tree = DecisionTreeClassifier(max_depth=6, random_state=14)
    tree = tree.fit(X_tr, y_tr)
    feature_importances = pd.Series(tree.feature_importances_,
                                    index = df_dec_tree.iloc[:,:-1].columns,
                                    name='Feature importance')\
                                    .sort_values(ascending=False)
    # plotting main feature importance
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    n_feat = feature_importances.shape[0]
    ax.set_ylabel('Feature importance', fontsize=12)
    ax.set_title('Main feature importance (Decision Tree)',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.bar(range(n_feat), feature_importances, color='b', edgecolor='k')
    plt.xticks(range(n_feat), feature_importances.index,
               ha='right', rotation=45, fontsize=12)
    plt.show()
    return n_clust