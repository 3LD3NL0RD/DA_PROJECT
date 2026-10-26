from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import chi2_contingency, chi2

BASE = Path(__file__).resolve().parent
OUT = BASE / 'analysis_outputs'
OUT.mkdir(exist_ok=True)
MIN_SUPPORT = .0025
MIN_CONFIDENCE = .10
TOP_K = 60
CHUNK = 1_000_000

def locate(*names, required=True):
    for name in names:
        p = BASE / name
        if p.exists(): return p
    if required: raise FileNotFoundError(f'Missing any of: {names}')
    return None

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close()

def main():
                                                     
    orders = pd.read_csv(locate('orders.csv'))
    products = pd.read_csv(locate('products(1).csv', 'products.csv'))
    aisles = pd.read_csv(locate('aisles.csv'))
    dept_path = locate('departments.csv', required=False)
    departments = pd.read_csv(dept_path) if dept_path else None
    prior_path = locate('order_products__prior.csv.zip', 'order_products__prior.csv')

    products = products.dropna(subset=['product_id', 'product_name']).drop_duplicates('product_id').copy()
    products['product_name'] = products['product_name'].astype(str).str.strip()
    aisles = aisles.drop_duplicates('aisle_id')
    if departments is not None: departments = departments.drop_duplicates('department_id')
    catalog = products.merge(aisles, on='aisle_id', how='left', validate='many_to_one')
    if departments is not None:
        catalog = catalog.merge(departments, on='department_id', how='left', validate='many_to_one')
    catalog.to_csv(OUT / 'cleaned_product_catalog.csv', index=False)
    name_of = catalog.set_index('product_id')['product_name'].to_dict()
    id_of = {v: k for k, v in name_of.items()}

    order_size = Counter()
    product_count = Counter()
    reordered_lines = 0
    total_lines = 0
    order_ids = set()
    dept_count = Counter()
    aisle_count = Counter()
    selected_pairs = [
        ('Organic Ginger Root', 'Organic Garlic'),
        ('Organic Cilantro', 'Limes'),
        ('Organic Garlic', 'Organic Yellow Onion'),
        ('Raspberries', 'Strawberries'),
        ('Limes', 'Large Lemon'),
        ('Organic Cilantro', 'Organic Red Onion'),
    ]
    needed_names = set(sum(([a, b] for a, b in selected_pairs), []))
    missing = needed_names - set(id_of)
    if missing: raise ValueError(f'Product names not found: {missing}')
    selected_ids = {id_of[n] for n in needed_names}
                                                                           
    for chunk in pd.read_csv(prior_path, chunksize=CHUNK, usecols=['order_id','product_id','reordered']):
        chunk = chunk.dropna(subset=['order_id','product_id']).drop_duplicates(['order_id','product_id'])
        total_lines += len(chunk)
        reordered_lines += int(chunk['reordered'].fillna(0).sum())
        sizes = chunk.groupby('order_id').size()
        order_size.update(sizes.to_dict())
        product_count.update(chunk['product_id'].value_counts().to_dict())
                                                                          
        c = chunk.merge(products[['product_id','department_id','aisle_id']], on='product_id', how='left')
        dept_count.update(c['department_id'].value_counts().to_dict())
        aisle_count.update(c['aisle_id'].value_counts().to_dict())
        del c

    N = len(order_size)
    basket_sizes = pd.Series(order_size, dtype='int64')
    assert int(basket_sizes.sum()) == total_lines
    top_ids = [int(x) for x, _ in product_count.most_common(TOP_K)]

    union_ids = sorted(set(top_ids) | selected_ids)
    index_of = {pid:i for i,pid in enumerate(union_ids)}
    order_to_row = {oid:i for i,oid in enumerate(order_size)}
    rows, cols = [], []
    for chunk in pd.read_csv(prior_path, chunksize=CHUNK, usecols=['order_id','product_id']):
        chunk = chunk[chunk['product_id'].isin(union_ids)].drop_duplicates(['order_id','product_id'])
        rows.extend(chunk['order_id'].map(order_to_row).to_numpy(dtype=np.int32))
        cols.extend(chunk['product_id'].map(index_of).to_numpy(dtype=np.int16))
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                               shape=(N, len(union_ids))).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1                                        
    del rows, cols
    counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
                                                          
    cross = (matrix.astype(np.int64).T @ matrix.astype(np.int64)).toarray()

    prior_orders = orders[orders['eval_set'].eq('prior')]
    print('Prior orders:', N, 'Prior item lines:', total_lines)
    print('Customers:', orders['user_id'].nunique(), 'Catalog products:', len(products))
    print('Mean/median basket size:', basket_sizes.mean(), basket_sizes.median())
    print('Reordered percentage:', 100*reordered_lines/total_lines)
    print('Average order number:', orders['order_number'].mean())
    print('Average days since prior:', orders['days_since_prior_order'].mean())
    print('Busiest day code:', orders['order_dow'].mode()[0])
    print('Busiest hour:', orders['order_hour_of_day'].mode()[0])
    top_products = pd.DataFrame([(pid, name_of.get(pid, '?'), cnt)
                                 for pid,cnt in product_count.most_common(20)],
                                columns=['product_id','product_name','prior_order_lines'])
    top_products.to_csv(OUT/'top_products.csv', index=False)
    dept_labels = (departments.set_index('department_id')['department'].to_dict()
                   if departments is not None else {})
    dept_table = pd.DataFrame([(k,dept_labels.get(k,str(k)),v,100*v/total_lines)
                               for k,v in dept_count.items()],
                              columns=['department_id','department','item_lines','percentage'])
    dept_table.sort_values('item_lines', ascending=False).to_csv(OUT/'department_counts.csv',index=False)
    aisle_labels = aisles.set_index('aisle_id')['aisle'].to_dict()
    pd.DataFrame([(k,aisle_labels.get(k,str(k)),v,100*v/total_lines) for k,v in aisle_count.items()],
                 columns=['aisle_id','aisle','item_lines','percentage']).sort_values('item_lines',ascending=False).to_csv(OUT/'aisle_counts.csv',index=False)
    plt.figure(figsize=(9,4.8))
    plt.hist(basket_sizes.clip(upper=50), bins=np.arange(1,53), edgecolor='white')
    plt.xlabel('Products per basket (50 = 50 or more)');plt.ylabel('Prior orders')
    plt.title('Prior basket size distribution');savefig('basket_size_histogram.png')
    plt.figure(figsize=(9,5))
    d = dept_table.nlargest(8,'item_lines').sort_values('item_lines')
    plt.barh(d['department'], d['item_lines']/1e6)
    plt.xlabel('Prior item lines (millions)');plt.title('Leading departments')
    savefig('department_chart.png')
    plt.figure(figsize=(10,5))
    p = top_products.head(10).iloc[::-1]
    plt.barh(p['product_name'],p['prior_order_lines'])
    plt.xlabel('Prior item lines');plt.title('Top 10 products')
    savefig('top_products_chart.png')

    top_cols = [index_of[pid] for pid in top_ids]
    frequent1 = [pid for pid in top_ids if counts[index_of[pid]]/N >= MIN_SUPPORT]
    frequent2 = []
    rules = []
    for i, a in enumerate(frequent1):
        for b in frequent1[i+1:]:
            n_ab = int(cross[index_of[a],index_of[b]])
            supp = n_ab/N
            if supp < MIN_SUPPORT: continue
            frequent2.append((a,b,n_ab,supp))
            for ant,cons in [(a,b),(b,a)]:
                conf = n_ab/counts[index_of[ant]]
                lift = conf/(counts[index_of[cons]]/N)
                if conf >= MIN_CONFIDENCE:
                    rules.append(dict(antecedent=name_of[ant],consequent=name_of[cons],
                                      both_orders=n_ab,support=supp,confidence=conf,lift=lift))
    pd.DataFrame([(name_of[a],int(counts[index_of[a]]),counts[index_of[a]]/N) for a in frequent1],
                 columns=['item','order_count','support']).to_csv(OUT/'apriori_frequent_1_itemsets.csv',index=False)
    pd.DataFrame([(name_of[a],name_of[b],n,s) for a,b,n,s in frequent2],
                 columns=['item_a','item_b','both_orders','support']).to_csv(OUT/'apriori_frequent_2_itemsets.csv',index=False)
    rule_df = pd.DataFrame(rules).sort_values('lift',ascending=False)
    rule_df.to_csv(OUT/'apriori_rules_top60.csv',index=False)

    verified = []
    for a,b in selected_pairs:
        ai,bi=index_of[id_of[a]],index_of[id_of[b]]
        n_a,n_b,n_ab=int(counts[ai]),int(counts[bi]),int(cross[ai,bi])
        verified.append(dict(antecedent=a,consequent=b,antecedent_orders=n_a,
                             consequent_orders=n_b,both_orders=n_ab,
                             support=n_ab/N,confidence=n_ab/n_a,
                             lift=(n_ab/N)/((n_a/N)*(n_b/N))))
    verified_df=pd.DataFrame(verified)
    verified_df.to_csv(OUT/'report_rules_verified.csv',index=False)
    print('\nVerified reported rules:\n',verified_df.to_string(index=False))
    plt.figure(figsize=(10,5))
    q=verified_df.sort_values('lift')
    labels=q['antecedent']+' → '+q['consequent']
    plt.barh(labels,q['lift'])
    plt.xlabel('Lift');plt.title('Selected association rules')
    savefig('selected_rule_lift.png')

    a,b='Organic Ginger Root','Organic Garlic'
    ai,bi=index_of[id_of[a]],index_of[id_of[b]]
    n_a,n_b,n_ab=int(counts[ai]),int(counts[bi]),int(cross[ai,bi])
    observed=np.array([[n_ab,n_a-n_ab],[n_b-n_ab,N-n_a-n_b+n_ab]],dtype=np.int64)
    chi2_stat,p_value,df,expected=chi2_contingency(observed,correction=False)
    critical=chi2.ppf(.95,df)
                                                                        
    decision='REJECT H0' if p_value<.05 else 'FAIL TO REJECT H0'
    print('\nObserved contingency table:\n',observed)
    print('Expected contingency table:\n',expected)
    print(f'Chi-square={chi2_stat:.6f}, df={df}, p={p_value:.6g}, critical={critical:.6f}')
    print('Decision:',decision)
    pd.DataFrame(observed,index=['Ginger present','Ginger absent'],
                 columns=['Garlic present','Garlic absent']).to_csv(OUT/'chi_square_observed.csv')
    pd.DataFrame(expected,index=['Ginger present','Ginger absent'],
                 columns=['Garlic present','Garlic absent']).to_csv(OUT/'chi_square_expected.csv')
    assert N==3214874 and n_ab==10792 and n_a==52087 and n_b==109778
    assert abs(chi2_stat-48070.492326676365)<1e-6
    summary=f'''Instacart Team 01 full prior-order analysis
Prior orders: {N:,}
Prior item lines: {total_lines:,}
Catalog products: {len(products):,}
Unique customers: {orders.user_id.nunique():,}
Average basket size: {basket_sizes.mean():.4f}
Median basket size: {basket_sizes.median()}
Reordered prior item lines: {100*reordered_lines/total_lines:.2f}%
H0: Ginger Root and Garlic presence are independent.
H1: Ginger Root and Garlic presence are associated.
Alpha: 0.05
Observed 2x2: {observed.tolist()}
Chi-square: {chi2_stat:.6f}
Degrees of freedom: {df}
P-value: <0.001 (numeric SciPy p={p_value})
Critical chi-square at alpha 0.05: {critical:.6f}
Decision: {decision}
Association does not imply causation.
'''
    (OUT/'summary.txt').write_text(summary,encoding='utf-8')
    print('\nSaved all results in',OUT)

if __name__=='__main__': main()
