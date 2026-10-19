"""Export the frozen method flow as editable vector/PDF and a preview PNG."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch


def render(output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,6.3)); ax.set_xlim(0,12); ax.set_ylim(0,6.3); ax.axis("off")
    dark="#24364b"; accent="#7044a0"; blue="#2d6f8a"
    def node(x,y,w,h,text,color=dark,fill="#f0f3f6"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07,rounding_size=0.08",facecolor=fill,edgecolor=color,lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10,color=color)
    def arrow(a,b,color=dark):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,lw=1.2,color=color,connectionstyle="arc3"))
    ax.text(.2,6.0,"MSSL V1  |  Training-only mask-supported side localization",fontsize=15,weight="bold",color=dark)
    node(.2,4.65,1.3,.7,"Image")
    node(2.1,4.65,2,.7,"Native Mask DINO")
    node(4.7,4.65,2.1,.7,"Native predictions\nclass / box / mask")
    node(7.5,4.65,1.8,.7,"Native losses\nretained")
    arrow((1.57,5),(2.03,5)); arrow((4.17,5),(4.63,5)); arrow((6.87,5),(7.43,5))
    node(.2,2.8,1.5,.65,"GT Mask",blue,"#eaf3f7")
    node(.2,1.7,1.5,.65,"Provided GT Box",blue,"#eaf3f7")
    node(2.25,2.2,2.2,1.0,"Axial foreground mass\n+ side-relative weights",accent,"#f2edf7")
    node(5.05,2.2,1.9,1.0,"Fixed smooth CDF\nfor each box side",accent,"#f2edf7")
    node(7.55,2.2,1.8,1.0,"MSSL loss\nfinal matched boxes",accent,"#f2edf7")
    node(10.05,3.5,1.6,.9,"Total loss\nLnative + λ Lmssl",accent,"#f2edf7")
    arrow((1.77,3.12),(2.18,2.95),blue); arrow((1.77,2.02),(2.18,2.42),blue)
    arrow((4.52,2.7),(4.98,2.7),accent); arrow((7.02,2.7),(7.48,2.7),accent)
    arrow((6.8,4.8),(8.42,3.27),accent)
    arrow((9.37,4.9),(10.4,4.47)); arrow((9.42,2.7),(10.4,3.43),accent)
    ax.text(5.5,1.6,"GT-only support; same box target; no new learned parameters",ha="center",fontsize=10,color=accent)
    ax.axhline(1.3,xmin=.015,xmax=.985,color="#bcc4ce",lw=1)
    ax.text(.2,.93,"INFERENCE",fontsize=11,weight="bold",color=dark)
    ax.text(2.0,.93,"Image  →  original Mask DINO  →  original class × predicted-mask-quality scoring  →  boxes",fontsize=10,color=dark)
    ax.text(2.0,.48,"No GT input. Predicted masks remain part of native inference. No additional inference module.",fontsize=9,color="#526171")
    fig.tight_layout(pad=.4)
    for suffix in ("svg","pdf","png"): fig.savefig(out/f"mssl_v1_pipeline.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); a=p.parse_args(); render(a.output)
