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

class CustTransformer(BaseEstimator) :

    def __init__(self, thresh_card=12,
                 strat_binary = 'ord', strat_low_card ='ohe',
                 strat_high_card ='bin', strat_quant = 'stand'):
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
        bin_cols = (X.nunique()[X.nunique()<=2].index)
        X_C_cols = X.select_dtypes(include=['object', 'category'])
        C_l_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique()\
                                    .between(3,self.thresh_card)].index
        C_h_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique()>self.thresh_card].index
        Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns\
                                                        if c not in bin_cols]
        d_t = {'binary': bin_cols,
               'low_card': C_l_card_cols,
               'high_card': C_h_card_cols,
               'numeric': Q_cols}
        d_t = {k:v for k,v in d_t.items() if len(v)}
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
            cols=None
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
                 'boxcox': PowerTransformer(method='boxcox'),
                 'yeo': PowerTransformer(method='yeo-johnson'),
                 'none': FunctionTransformer(func=lambda x:x,
                                             inverse_func=lambda x:x),
                 }

        # # dictionnaire liste des transfo categorielles EXISTANTES
        d_t = self.d_type_col(X)
        # numerics
        self.has_num = ('numeric' in d_t.keys())
        # categoricals
        self.has_cat = len([s for s in d_t.keys() if s in ['binary','low_card','high_card']])>0
        if self.has_cat:
            list_trans=[] # dictionnaire des transfo categorielles EXISTANTES
            for k, v in d_t.items():
                if k!='numeric':
                    list_trans.append((k,d_enc[self.dict_enc_strat[k]], v))
                    
            self.cat_cols = [] # liste des colonnes catégorielles à transformer
            for k,v in self.d_type_col(X).items():
                if k!='numeric': self.cat_cols += (list(v))
                
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

    def transform(self, X, y=None): # to get a dataframe
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
                    skip_outliers=True, thresh=3, layout=(3,3), save_enabled=False):

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

def plot_heatmap(corr, title, figsize=(8,4), vmin=-1, vmax=1, center=0,
                 palette = sns.color_palette("coolwarm", 20), shape='rect',
                 fmt='.2f', robust=False):
    
    fig, ax = plt.subplots(figsize=figsize)
    if shape == 'rect':
        mask=None
    elif shape == 'tri':
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        print('ERROR : this type of heatmap does not exist')

    palette = palette
    ax = sns.heatmap(corr, mask=mask, cmap=palette, vmin=vmin, vmax=vmax,
                     center=center, annot=True, annot_kws={"size": 10},fmt=fmt,
                     square=False, linewidths=.5, linecolor = 'white',
                     cbar_kws={"shrink": .9, 'label': None}, robust = robust,
                     xticklabels= corr.columns, yticklabels = corr.index)
    ax.tick_params(labelsize=10,top=False, bottom=True,
                labeltop=False, labelbottom=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",rotation_mode="anchor")
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


''' Plots the observations on two axis with clusters coloring if available (ser_clust)
tests for various transformations (quantile transformer)'''

from sklearn.preprocessing import QuantileTransformer


def plot_projection(df_2D, ser_clust=None, quant_transf=False,
                    title='', figsize=(5, 3), fig=None, ax=None):
    fig = plt.figure(figsize=figsize) if fig is None else fig
    colors = None if ser_clust is None else \
        [sns.color_palette()[x] for x in ser_clust.astype('int')]

    if quant_transf:

        ax = fig.add_subplot(131)
        ax.scatter(df_2D.iloc[:, 0], df_2D.iloc[:, 1],
                   s=1, alpha=0.7, c=colors)
        ax.set_title(title + ": no transf.")

        ax = fig.add_subplot(132)
        qt = QuantileTransformer(output_distribution='normal')
        df_2D_qtn = qt.fit_transform(df_2D)
        ax.scatter(df_2D_qtn[:, 0], df_2D_qtn[:, 1],
                   s=1, alpha=0.7, c=colors)
        ax.set_title(title + ": quantile trans. (normal)")

        ax = fig.add_subplot(133)
        qt = QuantileTransformer(output_distribution='uniform')
        df_2D_qtu = qt.fit_transform(df_2D)
        ax.scatter(df_2D_qtu[:, 0], df_2D_qtu[:, 1],
                   s=1, alpha=0.7, c=colors)
        ax.set_title(title + ": quantile trans. (uniform)")

    else:
        ax = fig.add_subplot(111) if ax is None else ax
        ax.scatter(df_2D.iloc[:, 0], df_2D.iloc[:, 1],
                   s=1, alpha=0.7, c=colors)
        ax.set_title(title + ": no transf.")

''' Computes Silhouette, Davies-Bouldin, Calinsky-Harabasz and inertia scores
for different number of clusters, then aggregate (mean std)
and plot the mean scores as a function of the number of clusters,
returns the aggregated scores as a dataframe'''

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def plot_clust_scores_vs_n_clust(df, n_clust=range(2,8),
                                 n_iter=10, figsize=(15,3)):

    silh_df, dav_bould_df, cal_harab_df, distor_df = \
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Looping on the number of clusters to compute the scores
    score_df_agg = pd.DataFrame()
    for i, n in enumerate(n_clust,1):
        silh, dav_bould, cal_harab, distor = [], [], [], []
        # Iterations of the same model (stability)
        for j in range(n_iter): 
            km = KMeans(n_clusters=n, n_jobs=-1)
            km.fit(df)
            ser_clust = pd.Series(data=km.labels_, index=df.index)
            # Computing scores for iterations
            silh.append(silhouette_score(X=df,
                                        labels=ser_clust))
            dav_bould.append(davies_bouldin_score(X=df,
                                        labels=ser_clust))
            cal_harab.append(calinski_harabasz_score(X=df,
                                        labels=ser_clust))
            distor.append(km.inertia_)                        
        # Dataframe of the results on iterations
        score_df = pd.DataFrame({'Silhouette': silh,
                                 'Davies_Bouldin': dav_bould,
                                 'Calinsky_Harabasz': cal_harab,
                                 'Distortion': distor})
        # --- Aggregation of the results
        ser_scores = score_df.agg(['mean', 'median', 'std']).unstack()\
                        .rename(n)
        score_df_agg = pd.concat([score_df_agg, ser_scores], axis=1)

    def gen_li(name): return [(name, s) for s in ('mean', 'median', 'std')]
    silh_df = score_df_agg.loc[gen_li('Silhouette')]
    dav_bould_df = score_df_agg.loc[gen_li('Davies_Bouldin')]
    cal_harab_df = score_df_agg.loc[gen_li('Calinsky_Harabasz')]
    distor_df = score_df_agg.loc[gen_li('Distortion')]

    # --- Plotting the results

    fig = plt.figure(figsize=figsize)

    def score_plot_vs_nb_clust(score_df, name, ax, c=None):
        score_df.T[(name,'mean')].plot(yerr=score_df.T[(name,'std')], elinewidth=1,
                            capsize=2, capthick=1, ecolor='k', fmt='-o',
                            c=c, ms=5, barsabove=False, uplims=False, ax=ax)
        
    ax = fig.add_subplot(141)
    score_plot_vs_nb_clust(silh_df,'Silhouette', ax, c='r')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette score')

    ax = fig.add_subplot(142)
    score_plot_vs_nb_clust(dav_bould_df,'Davies_Bouldin', ax, c='b')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Davies_Bouldin score')

    ax = fig.add_subplot(143)
    score_plot_vs_nb_clust(cal_harab_df,'Calinsky_Harabasz', ax, c='purple')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Calinsky_Harabasz score')

    ax = fig.add_subplot(144)
    score_plot_vs_nb_clust(distor_df,'Distortion', ax, c='g')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Distortion')

    fig.suptitle('Clustering score vs. number of clusters',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    return score_df_agg


''' Plots on the left the silhouette scores of the clusters and
on the right the projection of the points with cluster labels as cluster'''

from sklearn.metrics import silhouette_score, silhouette_samples

def silh_scores_vs_n_clust(df, n_clust, axis_2D, title='',
                           xlim=(-0.1, 0.8), figsize=(18, 3)):
    palette = sns.color_palette("husl", max(n_clust))
    colors_2 = palette.as_hex()

    distor = []
    for n in n_clust:
        fig = plt.figure(1, figsize=figsize)

        # --- Plot 1: Silhouette scores
        ax1 = fig.add_subplot(121)

        km = KMeans(n_clusters=n, random_state=14)
        ser_clust = km.fit_predict(df)
        distor.append(km.inertia_)

        sample_silh_val = silhouette_samples(df, ser_clust)

        y_lower = 10
        for i in range(n):
            # Aggregate and sort silh scores for samples of clust i
            ith_clust_silh_val = sample_silh_val[ser_clust == i]
            ith_clust_silh_val.sort()
            size_cluster_i = ith_clust_silh_val.shape[0]
            y_upper = y_lower + size_cluster_i
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0,
                              ith_clust_silh_val,
                              facecolor=colors_2[i],
                              edgecolor=colors_2[i],
                              alpha=0.7)

            # Label of silhouette plots with their clust. nb. at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Computes the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        silhouette_avg = silhouette_score(df, ser_clust)
        ax1.set_title("Nb of clusters={}, avg silhouhette score: {:.3f}" \
                      .format(n, silhouette_avg), fontsize=12)
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_xlim(list(xlim))
        # (n+1)*10: inserting blank spaces between clusters silh scores
        ax1.set_ylim([0, df.shape[0] + (n + 1) * 10])

        # --- Plot 1: Showing clusters on chosen projection
        ax2 = fig.add_subplot(122)

        colors = None if ser_clust is None else \
            [sns.color_palette()[x] for x in ser_clust.astype('int')]

        ax2.scatter(axis_2D.iloc[:, 0], axis_2D.iloc[:, 1],
                    s=1, alpha=0.7, c=colors)
        ax2.set_title(title, fontsize=12)

        ### A REPARER --- TRACAGE DES CENTROIDES SUR LES GRAPHES A INCLURE dans la fonction plot_projection
        # # Showing centers of the clusters
        # for i in n_clust:
        #     centers = km.cluster_centers_
        #     ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
        #                 marker='o', c="white", alpha=1, s=200, edgecolor='k')
        #     for i, c in enumerate(centers):
        #         ax2.scatter(c[0], c[1], c[2], marker='$%d$' % i,
        #                     alpha=1, s=50, edgecolor='k')
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


'''Computes n_iter times the clusters (various random initialisation) then
returns the mean and std of ARI between each unique couple of results
2 modes:
- 'init': initialisation stability: checks for stability of clusters for various
random initialisation of the clustering model
- 'samp': sampling stability: checks for stability of the clusters for differents
sampling of the training set'''

from sklearn.metrics import adjusted_rand_score

def stability(model, df, n_iter=5, mode='init', n_samp=10000):

    if mode == 'init':
        multiple_ser_clust = []
        for i in range(n_iter):
            model.fit(df)
            multiple_ser_clust.append(model.labels_)
        print("--- Testing for initialisation stability \
({} iterations) ---".format(n_iter))
    elif mode == 'samp':
        multiple_ser_clust = []
        for i in range(n_iter):
            df_samp = df.sample(n_samp)
            model.fit(df)
            multiple_ser_clust.append(model.labels_)
        print("--- Testing for stability with {} lines random samples \
(training set) ({} iterations) ---".format(n_samp, n_iter))


    # Computes ARI scores for each pair of models
    ARI_scores = []
    pairs_list = combinlist(np.arange(n_iter),2)
    for i, j in pairs_list:
        ARI_scores.append(adjusted_rand_score(multiple_ser_clust[i],
                                              multiple_ser_clust[j]))

    # Compute the mean and standard deviation of ARI scores
    ARI_mean, ARI_std = np.mean(ARI_scores), np.std(ARI_scores)
    print("Evaluation of stability with random init :\n\
        mean:{:.1f} , std: {:.1f} ".format(ARI_mean, ARI_std))

    return ARI_scores