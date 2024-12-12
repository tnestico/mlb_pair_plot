import seaborn as sns
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import requests
import polars as pl
from datetime import date
import api_scraper
import pandas as pd
import numpy as np
from  matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


from datasets import load_dataset
dataset = load_dataset('nesticot/mlb_data', data_files=['mlb_pitch_data_2024.csv' ])
dataset_train = dataset['train']
df = dataset_train.to_pandas().set_index(list(dataset_train.features.keys())[0]).reset_index(drop=True)

#df = pl.read_csv('mlb_pitch_data_2024.csv')

df_player = df.with_columns(
    (pl.concat_str(["pitcher_name", "pitcher_id"], separator=" - ").alias("pitcher_name_id"))
    )

# Select specific columns and convert to dictionary
pitcher_name_id_dict = dict(df_player.select(['pitcher_name_id', 'pitcher_id']).iter_rows())
# Display a selectbox for pitcher selection
selected_pitcher = st.selectbox("##### Select Pitcher", list(pitcher_name_id_dict.keys()))
pitcher_id = pitcher_name_id_dict[selected_pitcher]
vars = ['ivb','hb','plate_time']
df_plot = df.filter(pl.col('pitcher_id')==pitcher_id).drop_nulls(subset=['pitch_type']+vars).with_columns(
                pl.col('pitch_type').count().over('pitch_type').alias('pitch_count'))
pitch_type_list = df_plot['pitch_type'].unique().to_list()


### PITCH COLOURS ###

# Dictionary to map pitch types to their corresponding colors and names
pitch_colours = {
    ## Fastballs ##
    'FF': {'colour': '#FF007D', 'name': '4-Seam Fastball'},
    'FA': {'colour': '#FF007D', 'name': 'Fastball'},
    'SI': {'colour': '#98165D', 'name': 'Sinker'},
    'FC': {'colour': '#BE5FA0', 'name': 'Cutter'},

    ## Offspeed ##
    'CH': {'colour': '#F79E70', 'name': 'Changeup'},
    'FS': {'colour': '#FE6100', 'name': 'Splitter'},
    'SC': {'colour': '#F08223', 'name': 'Screwball'},
    'FO': {'colour': '#FFB000', 'name': 'Forkball'},

    ## Sliders ##
    'SL': {'colour': '#67E18D', 'name': 'Slider'},
    'ST': {'colour': '#1BB999', 'name': 'Sweeper'},
    'SV': {'colour': '#376748', 'name': 'Slurve'},

    ## Curveballs ##
    'KC': {'colour': '#311D8B', 'name': 'Knuckle Curve'},
    'CU': {'colour': '#3025CE', 'name': 'Curveball'},
    'CS': {'colour': '#274BFC', 'name': 'Slow Curve'},
    'EP': {'colour': '#648FFF', 'name': 'Eephus'},

    ## Others ##
    'KN': {'colour': '#867A08', 'name': 'Knuckleball'},
    'KN': {'colour': '#867A08', 'name': 'Knuckle Ball'},
    'PO': {'colour': '#472C30', 'name': 'Pitch Out'},
    'UN': {'colour': '#9C8975', 'name': 'Unknown'},
}

# Create dictionaries for pitch types and their attributes
dict_colour = {key: value['colour'] for key, value in pitch_colours.items()}
dict_pitch = {key: value['name'] for key, value in pitch_colours.items()}
dict_pitch_desc_type = {value['name']: key for key, value in pitch_colours.items()}
dict_pitch_desc_type.update({'Four-Seam Fastball':'FF'})
dict_pitch_desc_type.update({'All':'All'})
dict_pitch_name = {value['name']: value['colour'] for key, value in pitch_colours.items()}
dict_pitch_name.update({'Four-Seam Fastball':'#FF007D'})

font_properties = {'family': 'calibi', 'size': 12}
font_properties_titles = {'family': 'calibi', 'size': 20}
font_properties_axes = {'family': 'calibi', 'size': 16}

### PITCH ELLIPSE ###
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none',edgecolor='non', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")
    try:
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor,linewidth=2,linestyle='--',edgecolor=edgecolor, **kwargs)
        

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = x.mean()
        

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = y.mean()
        

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        

        ellipse.set_transform(transf + ax.transData)
    except ValueError:
         return    
        
    return ax.add_patch(ellipse)     

def plot_function():

    fig = plt.figure(figsize=(15,15),dpi=300)
    plt.rcParams.update({'figure.autolayout': True})
    fig.set_facecolor('white')
    sns.set_theme(style="whitegrid")

    plt_count = len(vars)
    gs = gridspec.GridSpec(plt_count+2, plt_count+2,
                        height_ratios=[1]+[10]*plt_count+[1],
                        width_ratios=[1]+[10]*plt_count+[1])

    for i in range(len(vars)):
        for j in range(len(vars)):
            print(i,j)
            ax = fig.add_subplot(gs[i+1,j+1])
            if i == j:
                sns.kdeplot(data=df_plot,x=vars[i], 
                            ax=ax,hue='pitch_type',palette=dict_colour,fill=True,common_norm=False,common_grid=True,linewidth=2)
                sns.kdeplot(data=df.filter((pl.col('pitch_type').is_in(pitch_type_list))&
                                        (pl.col('pitcher_hand')==df_plot['pitcher_hand'][0])),x=vars[i], 
                            ax=ax,hue='pitch_type',palette=dict_colour,fill=False,linestyle='--',common_norm=False,common_grid=True,linewidth=2)
                ax.get_legend().remove()
                if vars[j] == 'hb':
                    ax.set_xlim(-25,25)
                    ax.set_xlabel('Horizontal Break (in)')
                elif vars[j] == 'ivb':
                    ax.set_xlim(-25,25)   
                    ax.set_xlabel('Vertical Break (in)')
                else:
                    ax.set_xlim(0.35,0.55)
                    ax.set_xlabel('Time to Plate (s)')

            elif i < j:
                sns.scatterplot(data=df_plot, x=vars[j], y=vars[i], ax=ax,hue='pitch_type',palette=dict_colour)
                for pitch in pitch_type_list:
                    confidence_ellipse(df.filter((pl.col('pitch_type')==pitch)&
                                        (pl.col('pitcher_hand')==df_plot['pitcher_hand'][0]))[vars[j]].to_numpy(),
                                        df.filter((pl.col('pitch_type')==pitch)&
                                        (pl.col('pitcher_hand')==df_plot['pitcher_hand'][0]))[vars[i]].to_numpy(),ax,n_std=1.5,edgecolor=dict_colour[pitch], alpha=0.4)
                ax.get_legend().remove()
                if vars[j] == 'hb':
                    ax.set_xlim(-25,25)
                    ax.set_xlabel('Horizontal Break (in)')
                elif vars[j] == 'ivb':
                    ax.set_xlim(-25,25)   
                    ax.set_xlabel('Induced Vertical Break (in)')
                else:
                    ax.set_xlim(0.35,0.55)
                    ax.set_xlabel('Time to Plate (s)')

                if vars[i] == 'hb':
                    ax.set_ylim(-25,25)
                    ax.set_ylabel('Horizontal Break (in)')
                elif vars[i] == 'ivb':
                    ax.set_ylim(-25,25)
                    ax.set_ylabel('Induced Vertical Break (in)')
                else:
                    ax.set_ylim(0.35,0.55)
                    ax.set_ylabel('Time to Plate (s)')
            elif j==0 and i==2:

                solid_line = Line2D([0], [0], color='black', lw=2, linestyle='-')
                dashed_line = Line2D([0], [0], color='black', lw=2, linestyle='--')
                # Create legend for pitch types
                items_in_order = (df_plot.sort("pitch_count", descending=True)['pitch_type'].unique(maintain_order=True).to_numpy())
                colour_pitches = [dict_colour[x] for x in items_in_order]
                label = [dict_pitch[x] for x in items_in_order]
                handles = [plt.scatter([], [], color=color, marker='o', s=100) for color in colour_pitches]
                if len(label) > 5:
                    ax.legend(handles+[solid_line, dashed_line], label+[f'{df_plot["pitcher_name"][0]}', f'MLB {df_plot["pitcher_hand"][0]}HP Average'], bbox_to_anchor=(0.2, 0.2, 0.6, 0.6), ncol=1,
                            fancybox=True, loc='upper center', fontsize=16, framealpha=1.0, markerscale=1.2)
                else:
                    ax.legend(handles+[solid_line, dashed_line], label+[f'{df_plot["pitcher_name"][0]}', f'MLB {df_plot["pitcher_hand"][0]}HP Average'], bbox_to_anchor=(0.2, 0.2, 0.6, 0.6), ncol=1,
                            fancybox=True, loc='upper center', fontsize=16, framealpha=1.0, markerscale=1.2)
                ax.axis('off')            
                # Create custom legend handles


            else:
                ax.axis('off')
            

    ax_top = fig.add_subplot(gs[0,:])
    ax_top.axis('off')
    ax_bottom = fig.add_subplot(gs[-1,:])
    ax_bottom.axis('off')
    ax_left = fig.add_subplot(gs[:,0])
    ax_left.axis('off')
    ax_right = fig.add_subplot(gs[:,-1])
    ax_right.axis('off')

    # Add labels to the top
    ax_top.text(0.5,0,f'{df_plot["pitcher_name"][0]} - {df_plot["pitcher_hand"][0]}HP - 2024 MLB Season',fontsize=24,ha='center',va='top')

    # Add labels to the top
    ax_bottom.text(0.1,0.5,'By: @TJStats',fontsize=20,ha='left',va='bottom')
    # Add labels to the top
    ax_bottom.text(0.9,0.5,'Data: MLB',fontsize=20,ha='right',va='bottom')

    st.pyplot(fig)

# Button to generate plot
if st.button('Generate Plot'):
    try:
        plot_function()
    except IndexError:
        st.write('Please select different parameters.')