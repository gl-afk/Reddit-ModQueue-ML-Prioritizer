import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mpl

df = pd.read_csv('final_intersection_dataset.csv')

# biggest subreddit
top_sub = df['community'].value_counts().index[0]
print(f"Analyzing subreddit: {top_sub}")
df_sub = df[df['community'] == top_sub]

# total votes and mod label per post
posts = df_sub.groupby('item_id').agg(
    total_upvotes=('vote', lambda x: (x == 1).sum()),
    total_downvotes=('vote', lambda x: (x == -1).sum()),
    mod_label=('label', 'first')
).reset_index()

# correlation matrix
print("\nCorrelation Matrix")
print(posts[['total_upvotes', 'total_downvotes', 'mod_label']].corr())

# facetgrid
g = sb.FacetGrid(posts, col="mod_label", height=5, aspect=1.2) # one for approved posts, the other for removed posts
g.map(sb.scatterplot, "total_downvotes", "total_upvotes", alpha=0.6, color="#2ecc71")

g.set_axis_labels("Downvotes", "Upvotes")
g.set_titles(col_template="Mod Label: {col_name}")
mpl.subplots_adjust(top=0.85)
g.figure.suptitle(f"Community Signal Analysis in {top_sub}", fontsize=14, fontweight='bold')

mpl.show()