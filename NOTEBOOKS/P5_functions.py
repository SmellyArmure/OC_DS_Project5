# Printing total nb and percentage of null

def print_null_pct(df):
    tot_null = df.isna().sum().sum()
    print('nb of null: ', tot_null, '\npct of null: ',
        '{:.1f}'.format(tot_null*100/(df.shape[0]*df.shape[1])))

# Displaying number of missing values per column

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

def move_cat_containing(my_index, strings, order='last'):
	idx_sel = []
	if order == 'last':
	    index = my_index
	elif order == 'first':
	    index = my_index[::-1]
	else:
	    print("--- WARNING : index unchanged.\n -- Wrong order passed. Pass 'first' or 'last'")
	    return my_index
	for s in strings:
	    idx_sel += [i for i,x in enumerate(index) if x in index[index.str.contains(s)]]
	to_move = index[idx_sel]
	rank = max(idx_sel)
	mod_index = index.drop(to_move)
	for c in to_move:
	    mod_index = mod_index.insert(rank,c)
	return mod_index if order=='last' else mod_index[::-1]