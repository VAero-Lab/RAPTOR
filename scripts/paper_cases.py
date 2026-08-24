#!/usr/bin/env python3
"""
CONDOR Paper — Generate All Figures (v2)
==========================================
Usage:
    python generate_paper_figures.py                                # quick test
    python generate_paper_figures.py --maxiter 200 --popsize 20     # publication
"""
import sys,os,argparse,time,numpy as np
from dataclasses import dataclass
from typing import List
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
try:
    from raptor.dem import DEMInterface;from raptor.config import UAVConfig,MissionConstraints
    from raptor.builder import FacilityNode,PathBuilder,PathStrategy
    from raptor.routed_path import RoutedPath;from raptor.energy import AircraftEnergyParams,analyze_path_energy,power_fw_cruise,power_hover
    from raptor.terrain import TerrainAnalyzer;from raptor.airspace import build_airspace,CircularZone,PolygonalZone
    from raptor.optimizer import PathOptimizer,OptMode;from raptor.astar_baseline import AStarGridPlanner
except ImportError:
    from uav_path_planning.dem import DEMInterface;from uav_path_planning.config import UAVConfig,MissionConstraints
    from uav_path_planning.builder import FacilityNode,PathBuilder,PathStrategy
    from uav_path_planning.routed_path import RoutedPath;from uav_path_planning.energy import AircraftEnergyParams,analyze_path_energy,power_fw_cruise,power_hover
    from uav_path_planning.terrain import TerrainAnalyzer;from uav_path_planning.airspace import build_airspace,CircularZone,PolygonalZone
    from uav_path_planning.optimizer import PathOptimizer,OptMode;from uav_path_planning.astar_baseline import AStarGridPlanner
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt;import matplotlib.patheffects as pe
from matplotlib.colors import LightSource,LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

# ═══ STYLE ═══
_TCMAP=LinearSegmentedColormap.from_list('mt',['#C8B99A','#A8B89A','#8A9A7A','#7A8A6A','#8A8A7A','#AAAAAA','#CCCCCC','#E0E0E0'])
C_STR='#616161';C_AST='#F57F17';C_OPT='#1565C0';C_RDAC='#C62828'
ZC={'prohibited':('#B71C1C',0.25),'restricted':('#E65100',0.18),'aerodrome_ctr':('#C62828',0.12),'populated_area':('#EF6C00',0.18),'ecological':('#2E7D32',0.22),'altitude_limit':('#FBC02D',0.03)}

def set_style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif','serif'],'font.size':9,'axes.labelsize':10,'axes.titlesize':10,'axes.titleweight':'bold','axes.linewidth':0.5,'axes.grid':False,'xtick.labelsize':8,'ytick.labelsize':8,'xtick.direction':'in','ytick.direction':'in','xtick.major.width':0.4,'ytick.major.width':0.4,'xtick.major.size':3,'ytick.major.size':3,'legend.fontsize':7,'legend.framealpha':0.92,'legend.edgecolor':'0.75','figure.dpi':200,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.04,'lines.linewidth':1.4,'patch.linewidth':0.4})

# ═══ SCENARIOS ═══
@dataclass
class Leg:
    origin:FacilityNode;destination:FacilityNode;payload_kg:float;description:str
@dataclass
class Scenario:
    sid:str;name:str;urgency:str;legs:List[Leg];color:str

def build_scenarios(dem):
    def F(n,la,lo):return FacilityNode(n,la,lo,dem.elevation(la,lo))
    HG=F('H.Garcés',-0.2444,-78.5411);HE=F('H.Espejo',-0.2144,-78.4987);HP=F('H.P.A.Suárez',-0.1273,-78.4977)
    CL=F('CS.Lloa',-0.2477,-78.5803);CT=F('CS.Tumbaco',-0.2158,-78.4085);CC=F('CS.CentroHist',-0.2231,-78.5145)
    CG=F('CS.Guamaní',-0.3242,-78.5493);CA=F('CS.Calacalí',0.0000,-78.5151);CP=F('CS.Pintag',-0.3755,-78.3733)
    CV=F('CS.Vicentina',-0.2179,-78.4848);CB=F('CS.Chimbacalle',-0.2447,-78.5137)
    return [
        Scenario('S1','Garcés → Lloa','routine',[Leg(HG,CL,2.0,'Vaccines'),Leg(CL,HG,0.0,'Return')],'#2196F3'),
        Scenario('S2','Espejo → Tumbaco','urgent',[Leg(HE,CT,0.0,'Empty'),Leg(CT,HE,1.5,'Samples')],'#FF9800'),
        Scenario('S3','Espejo → Centro Hist.','urgent',[Leg(HE,CC,1.0,'Emergency meds'),Leg(CC,HE,0.0,'Return')],'#F44336'),
        Scenario('S4','Garcés → Guamaní','routine',[Leg(HG,CG,2.0,'Medicines'),Leg(CG,HG,1.5,'Samples')],'#4CAF50'),
        Scenario('S5','P.A.Suárez → Calacalí','routine',[Leg(HP,CA,2.5,'Vaccines'),Leg(CA,HP,0.0,'Return')],'#9C27B0'),
        Scenario('S6','Garcés → Pintag','emergency',[Leg(HG,CP,2.5,'Blood'),Leg(CP,HG,2.0,'Samples')],'#D32F2F'),
        Scenario('S7','Espejo urban tour','routine',[Leg(HE,CV,2.0,'Meds'),Leg(CV,CB,1.0,'Partial'),Leg(CB,HE,1.5,'Samples')],'#00BCD4'),
        Scenario('S8','Garcés southern tour','emergency',[Leg(HG,CG,2.0,'Supplies'),Leg(CG,CP,1.0,'Partial'),Leg(CP,HG,0.0,'Return')],'#795548'),
    ]

# ═══ DRAWING ═══
def draw_dem(ax,dem,lr,lo):
    lm=(dem.lat_1d>=lr[0])&(dem.lat_1d<=lr[1]);lnm=(dem.lon_1d>=lo[0])&(dem.lon_1d<=lo[1])
    ls_,lo_=dem.lat_1d[lm],dem.lon_1d[lnm];el=dem.elev_grid[np.ix_(lm,lnm)]
    LO,LA=np.meshgrid(lo_,ls_);li=LightSource(azdeg=315,altdeg=40)
    rgb=li.shade(el,cmap=_TCMAP,blend_mode='soft',vmin=np.nanmin(el)-100,vmax=np.nanmax(el)+100)
    ax.imshow(rgb,extent=[lo_[0],lo_[-1],ls_[0],ls_[-1]],origin='lower',aspect='auto')
    cs=ax.contour(LO,LA,el,levels=np.arange(np.nanmin(el)//200*200,np.nanmax(el)+300,200),colors='0.45',linewidths=0.2,alpha=0.5)
    ax.clabel(cs,inline=True,fontsize=4.5,fmt='%d')

def draw_zones(ax,airspace,lr,lo):
    for z in airspace.active_zones():
        if z.is_global:continue
        zc,za=ZC.get(z.zone_type.value,('#999',0.1))
        if isinstance(z.geometry,CircularZone):
            cz=z.geometry;rl=cz.radius_m/111320.0;rn=cz.radius_m/(111320.0*np.cos(np.radians(cz.center_lat)))
            t=np.linspace(0,2*np.pi,80);ax.fill(cz.center_lon+rn*np.cos(t),cz.center_lat+rl*np.sin(t),color=zc,alpha=za,zorder=2)
            ax.plot(cz.center_lon+rn*np.cos(t),cz.center_lat+rl*np.sin(t),'-',color=zc,lw=0.5,alpha=0.4,zorder=2)
            if lr[0]<cz.center_lat<lr[1] and lo[0]<cz.center_lon<lo[1]:
                ax.text(cz.center_lon,cz.center_lat,z.zone_id.split('_')[-1],ha='center',va='center',fontsize=4.5,color=zc,fontweight='bold',path_effects=[pe.withStroke(linewidth=1.5,foreground='white')])
        elif isinstance(z.geometry,PolygonalZone):
            vs=[(lon,lat) for lat,lon in z.geometry.vertices];ax.add_patch(MplPolygon(vs,closed=True,fc=zc,alpha=za,ec=zc,lw=0.5,zorder=2))

def path_arrowed(ax,fp,color,label=None,lw=1.8,ls='-',zorder=5):
    wp=fp.get_waypoints_array()
    ax.plot(wp[:,1],wp[:,0],ls,color=color,lw=lw,alpha=0.9,zorder=zorder,label=label,path_effects=[pe.withStroke(linewidth=lw+1.2,foreground='white')])
    n=len(wp)
    for frac in [0.3,0.6,0.85]:
        idx=min(int(frac*n),n-2);dx=wp[idx+1,1]-wp[idx,1];dy=wp[idx+1,0]-wp[idx,0]
        if abs(dx)+abs(dy)>1e-8:
            ax.annotate('',xy=(wp[idx+1,1],wp[idx+1,0]),xytext=(wp[idx,1],wp[idx,0]),arrowprops=dict(arrowstyle='->',color=color,lw=lw*0.7),zorder=zorder+1)

def astar_plan(ax,wps,color=C_AST,label='A* grid',lw=1.0):
    if not wps:return
    ax.plot([w[1] for w in wps],[w[0] for w in wps],'--',color=color,lw=lw,alpha=0.8,zorder=4,label=label,path_effects=[pe.withStroke(linewidth=lw+0.8,foreground='white')])

def draw_profile(ax,dem,fp,color,label=None,lw=1.8,ls='-',ceiling=False):
    wp=fp.get_waypoints_array();dk=wp[:,4]/1000;terr=dem.elevation_batch(wp[:,0],wp[:,1]);v=~np.isnan(terr)
    ax.fill_between(dk[v],terr[v],alpha=0.18,color='#8D6E63',lw=0);ax.plot(dk[v],terr[v],'-',color='#6D4C41',lw=0.5,label='Terrain')
    if ceiling:c=terr.copy();c[v]+=120;ax.plot(dk[v],c[v],'--',color='#BF360C',lw=0.7,alpha=0.6,label='120m AGL')
    ax.plot(dk,wp[:,2],ls,color=color,lw=lw,label=label,path_effects=[pe.withStroke(linewidth=lw+0.8,foreground='white')])

def astar_profile(ax,wps,dem,color=C_AST,label='A* grid',lw=1.0):
    if not wps:return
    ds=[0];
    for i in range(1,len(wps)):ds.append(ds[-1]+DEMInterface.haversine(wps[i-1][0],wps[i-1][1],wps[i][0],wps[i][1])/1000)
    ax.plot(ds,[w[2] for w in wps],'--',color=color,lw=lw,label=label,alpha=0.8)

def endpts(ax,o,d,fs=5.5):
    ax.plot(o.lon,o.lat,'o',color='#2E7D32',ms=7,mec='k',mew=0.6,zorder=8)
    ax.plot(d.lon,d.lat,'s',color='#C62828',ms=7,mec='k',mew=0.6,zorder=8)
    for f,ofs,c in [(o,(-5,5),'#1B5E20'),(d,(5,-5),'#B71C1C')]:
        ax.annotate(f.name,(f.lon,f.lat),xytext=ofs,textcoords='offset points',fontsize=fs,fontweight='bold',color=c,bbox=dict(boxstyle='round,pad=0.12',fc='white',alpha=0.9,ec='0.6',lw=0.3))

def cax(ax,xl='',yl=''):ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);xl and ax.set_xlabel(xl);yl and ax.set_ylabel(yl)

def ibox(ax,t):ax.text(0.03,0.97,t,transform=ax.transAxes,fontsize=6,va='top',fontfamily='monospace',bbox=dict(boxstyle='round,pad=0.25',fc='white',alpha=0.92,ec='0.65',lw=0.3))

def flbl(ax,f):
    l='FEASIBLE' if f else 'INFEASIBLE';c='#2E7D32' if f else '#C62828'
    ax.text(0.97,0.03,l,transform=ax.transAxes,fontsize=6.5,va='bottom',ha='right',fontweight='bold',color=c,bbox=dict(boxstyle='round,pad=0.15',fc='white',alpha=0.9,ec=c,lw=0.6))

# ═══ RUN ═══
@dataclass
class LR:
    rp:object;e_res:object;ar_res:object;opt_result:object;soc_start:float;soc_end:float;fp_straight:object;astar_wps:list

def run_sc(sc,dem,uav,con,ac,opt,bld,ast,air,MI,POP,SD,nw=1,bwh=600):
    rs=[];soc=1.0
    for i,leg in enumerate(sc.legs):
        o,d=leg.origin,leg.destination
        fps=bld.build(o,d,PathStrategy.HIGH_OVERFLY)
        try:ar=ast.plan((o.lat,o.lon,o.ground_elev),(d.lat,d.lon,d.ground_elev));awps=ar.waypoints if ar.path_found else []
        except:awps=[]
        rp=RoutedPath(o,d,dem,uav,con,n_intermediate=nw)
        optr=opt.optimize_routed(rp,mode=OptMode.ENERGY,payload_kg=leg.payload_kg,airspace=air,maxiter=MI,popsize=POP,verbose=False,seed=SD+i)
        er=analyze_path_energy(rp.flight_path,ac);arr=air.check_path(rp.flight_path) if air else None
        ss=soc;se=max(soc-er.total_energy_wh/bwh,0.0);rs.append(LR(rp,er,arr,optr,ss,se,fps,awps));soc=se
    return rs

# ═══ FIGURES ═══
def fig1(dem,air,scs,out):
    fig,ax=plt.subplots(figsize=(6.5,8));lr,lo=(-0.40,0.08),(-78.65,-78.33)
    draw_dem(ax,dem,lr,lo);draw_zones(ax,air,lr,lo)
    done=set()
    for sc in scs:
        for lg in sc.legs:
            ax.plot([lg.origin.lon,lg.destination.lon],[lg.origin.lat,lg.destination.lat],'-',color=sc.color,lw=0.7,alpha=0.4,zorder=3)
            for f in [lg.origin,lg.destination]:
                k=(round(f.lat,4),round(f.lon,4))
                if k not in done:
                    ih='H.' in f.name;ax.plot(f.lon,f.lat,'o' if ih else 's',color='#1565C0' if ih else '#E65100',ms=5 if ih else 3.5,mec='k',mew=0.4,zorder=8)
                    ax.annotate(f.name,(f.lon,f.lat),xytext=(3,2),textcoords='offset points',fontsize=3.5,color='0.15',bbox=dict(boxstyle='round,pad=0.08',fc='white',alpha=0.8,ec='none'));done.add(k)
    ax.set_xlim(*lo);ax.set_ylim(*lr);ax.set_xlabel('Longitude [°]');ax.set_ylabel('Latitude [°]')
    ax.set_title('Quito DMQ — Medical Network and RDAC 101 Airspace');fig.savefig(f'{out}/fig01_study_area.png');plt.close(fig);print('  ✓ fig01')

def fig2(dem,air,bld,opt,ast,uav,con,ac,sc,MI,POP,SD,out):
    o,d,pay=sc.legs[0].origin,sc.legs[0].destination,sc.legs[0].payload_kg
    fps=bld.build(o,d,PathStrategy.HIGH_OVERFLY);es=analyze_path_energy(fps,ac);ars=air.check_path(fps)
    try:ar=ast.plan((o.lat,o.lon,o.ground_elev),(d.lat,d.lon,d.ground_elev));awps=ar.waypoints if ar.path_found else []
    except:awps=[];ar=None
    rpd=RoutedPath(o,d,dem,uav,con,n_intermediate=0);opt.optimize_routed(rpd,mode=OptMode.ENERGY,payload_kg=pay,airspace=None,maxiter=MI,popsize=POP,verbose=False,seed=SD)
    ed=analyze_path_energy(rpd.flight_path,ac);ard=air.check_path(rpd.flight_path)
    rpr=RoutedPath(o,d,dem,uav,con,n_intermediate=1);opt.optimize_routed(rpr,mode=OptMode.ENERGY,payload_kg=pay,airspace=air,maxiter=MI,popsize=POP,verbose=False,seed=SD)
    er=analyze_path_energy(rpr.flight_path,ac);arr=air.check_path(rpr.flight_path)
    afps=[fps,rpd.flight_path,rpr.flight_path];alat=np.concatenate([f.get_waypoints_array()[:,0] for f in afps]);alon=np.concatenate([f.get_waypoints_array()[:,1] for f in afps])
    if awps:alat=np.append(alat,[w[0] for w in awps]);alon=np.append(alon,[w[1] for w in awps])
    m=0.012;lr=(min(alat)-m,max(alat)+m);lo=(min(alon)-m,max(alon)+m)
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    cfgs=[('(a) Straight line',fps,None,C_STR,es,ars,False),('(b) A* grid',None,None,C_AST,None,None,False),('(c) DE optimized',rpd.flight_path,rpd,C_OPT,ed,ard,False),('(d) RDAC 101',rpr.flight_path,rpr,C_RDAC,er,arr,True)]
    for col,(lbl,fp,rp,clr,e,a,sz) in enumerate(cfgs):
        ax=axes[0,col];draw_dem(ax,dem,lr,lo)
        if sz:draw_zones(ax,air,lr,lo)
        ax.plot([o.lon,d.lon],[o.lat,d.lat],':',color='0.5',lw=0.5,zorder=3)
        if col==1:
            astar_plan(ax,awps);
            if ar and awps:ibox(ax,f"E≈{ar.total_energy_wh:.0f} Wh")
        else:
            path_arrowed(ax,fp,clr,lbl.split(') ')[1])
            if rp and hasattr(rp,'n_intermediate') and rp.n_intermediate>0:
                for lat,lon in rp.waypoint_positions[1:-1]:ax.plot(lon,lat,'D',color=clr,ms=4,mec='k',mew=0.4,zorder=6)
            ibox(ax,f"E={e.total_energy_wh:.0f} Wh\nt={e.total_time/60:.1f} min\nAirV={a.n_violations}");flbl(ax,a.feasible)
        if col==0:endpts(ax,o,d)
        ax.set_xlim(*lo);ax.set_ylim(*lr);ax.set_xlabel('Lon [°]')
        if col==0:ax.set_ylabel('Lat [°]')
        ax.set_title(lbl)
        ax2=axes[1,col]
        if col==1:astar_profile(ax2,awps,dem);
        else:draw_profile(ax2,dem,fp,clr,ceiling=sz)
        ax2.set_xlabel('Distance [km]')
        if col==0:ax2.set_ylabel('Altitude [m]')
        ax2.set_title(f'({chr(101+col)}) Profile');ax2.legend(fontsize=5.5,loc='best');cax(ax2)
    fig.suptitle(f'{sc.sid}: {sc.name} — Four-Path Comparison',fontsize=11,y=1.01);plt.tight_layout()
    fig.savefig(f'{out}/fig02_four_path_{sc.sid}.png');plt.close(fig);print(f'  ✓ fig02 ({sc.sid})')

def fig3(aoff,aon,scs,out):
    fig,(a1,a2)=plt.subplots(2,1,figsize=(10,7),height_ratios=[3,2])
    x=np.arange(len(scs));w=0.35
    eo=[sum(l.e_res.total_energy_wh for l in aoff[s.sid]) for s in scs];en=[sum(l.e_res.total_energy_wh for l in aon[s.sid]) for s in scs]
    so=[aoff[s.sid][-1].soc_end for s in scs];sn=[aon[s.sid][-1].soc_end for s in scs]
    uc={'routine':'#66BB6A','urgent':'#FFA726','emergency':'#EF5350'}
    a1.bar(x-w/2,eo,w,label='Airspace OFF',color='#BBDEFB',edgecolor='0.4',lw=0.3)
    a1.bar(x+w/2,en,w,label='Airspace ON',color=[uc[s.urgency] for s in scs],edgecolor='0.4',lw=0.3)
    a1.set_xticks(x);a1.set_xticklabels([s.sid for s in scs]);a1.set_ylabel('Energy [Wh]');a1.set_title('(a) Energy');a1.legend(fontsize=7);cax(a1)
    a2.bar(x-w/2,[s*100 for s in so],w,color='#BBDEFB',edgecolor='0.4',lw=0.3,label='SOC OFF')
    a2.bar(x+w/2,[s*100 for s in sn],w,color=[uc[s.urgency] for s in scs],edgecolor='0.4',lw=0.3,label='SOC ON')
    a2.axhline(y=15,color='#C62828',ls='--',lw=0.6);a2.set_xticks(x);a2.set_xticklabels([f'{s.sid}\n{s.name}' for s in scs],fontsize=6)
    a2.set_ylabel('Final SOC [%]');a2.set_title('(b) Battery SOC');a2.legend(fontsize=7);cax(a2);a2.set_ylim(0,105)
    plt.tight_layout();fig.savefig(f'{out}/fig03_scenarios.png');plt.close(fig);print('  ✓ fig03')

def fig4(dem,air,aon,scs,out):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,sid in enumerate(['S1','S6']):
        sc=[s for s in scs if s.sid==sid][0];lg0=aon[sid][0];o,d=sc.legs[0].origin,sc.legs[0].destination
        wp=lg0.rp.flight_path.get_waypoints_array();ws=lg0.fp_straight.get_waypoints_array()
        al=np.concatenate([wp[:,0],ws[:,0]]);an=np.concatenate([wp[:,1],ws[:,1]]);m=0.015
        lat_r=(min(al)-m,max(al)+m);lon_r=(min(an)-m,max(an)+m)
        ax=axes[0,col];draw_dem(ax,dem,lat_r,lon_r);draw_zones(ax,air,lat_r,lon_r)
        ax.plot([o.lon,d.lon],[o.lat,d.lat],':',color=C_STR,lw=0.6,zorder=3,label='Direct')
        astar_plan(ax,lg0.astar_wps);path_arrowed(ax,lg0.rp.flight_path,C_RDAC,'RDAC 101')
        endpts(ax,o,d);ax.set_xlim(*lon_r);ax.set_ylim(*lat_r)
        if col==0:ax.set_ylabel('Lat [°]')
        ax.set_xlabel('Lon [°]');lb='Terrain-dominated' if sid=='S1' else 'Zone-dominated'
        ax.set_title(f'({chr(97+col)}) {sc.sid}: {sc.name}');ax.legend(fontsize=5.5,loc='lower left')
        ax2=axes[1,col];draw_profile(ax2,dem,lg0.fp_straight,C_STR,'Straight',lw=0.8,ceiling=True)
        astar_profile(ax2,lg0.astar_wps,dem);draw_profile(ax2,dem,lg0.rp.flight_path,C_RDAC,'RDAC 101')
        if col==0:ax2.set_ylabel('Altitude [m]')
        ax2.set_xlabel('Distance [km]');ax2.set_title(f'({chr(99+col)}) Profile');ax2.legend(fontsize=5.5);cax(ax2)
    fig.suptitle('Corridor Reshaping: Terrain vs Regulatory Constraints',fontsize=11,y=1.01);plt.tight_layout()
    fig.savefig(f'{out}/fig04_topology.png');plt.close(fig);print('  ✓ fig04')

def fig5(dem,air,aon,scs,out):
    sid='S8';sc=[s for s in scs if s.sid==sid][0];legs=aon[sid];clrs=['#1565C0','#E65100','#2E7D32']
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
    al,an=[],[]
    for lg in legs:wp=lg.rp.flight_path.get_waypoints_array();al.extend(wp[:,0]);an.extend(wp[:,1])
    m=0.02;lat_r=(min(al)-m,max(al)+m);lon_r=(min(an)-m,max(an)+m)
    draw_dem(a1,dem,lat_r,lon_r);draw_zones(a1,air,lat_r,lon_r)
    for i,lg in enumerate(legs):
        path_arrowed(a1,lg.rp.flight_path,clrs[i%3],f'Leg {i+1}',lw=1.5)
        if i==0:astar_plan(a1,lg.astar_wps)
        o,d=sc.legs[i].origin,sc.legs[i].destination;a1.plot([o.lon,d.lon],[o.lat,d.lat],':',color='0.5',lw=0.4,zorder=3)
    for i,lg in enumerate(sc.legs):a1.plot(lg.origin.lon,lg.origin.lat,'o',color=clrs[i%3],ms=5,mec='k',mew=0.4,zorder=8)
    a1.set_xlim(*lon_r);a1.set_ylim(*lat_r);a1.set_xlabel('Lon [°]');a1.set_ylabel('Lat [°]');a1.set_title('(a) Flight corridors');a1.legend(fontsize=5.5)
    cd=0
    for i,lg in enumerate(legs):
        wp=lg.rp.flight_path.get_waypoints_array();ld=wp[-1,4]/1000;x0,x1=cd,cd+ld
        a2.fill_between([x0,x1],lg.soc_start,lg.soc_end,alpha=0.25,color=clrs[i%3],lw=0)
        a2.plot([x0,x1],[lg.soc_start,lg.soc_end],'-o',color=clrs[i%3],lw=2,ms=4,mec='k',mew=0.3)
        a2.text((x0+x1)/2,(lg.soc_start+lg.soc_end)/2,f"Leg {i+1}\n{lg.e_res.total_energy_wh:.0f} Wh\n{sc.legs[i].payload_kg:.1f} kg",ha='center',va='center',fontsize=5.5,bbox=dict(boxstyle='round,pad=0.15',fc='white',alpha=0.9,ec='0.7',lw=0.3))
        cd=x1
    a2.axhline(y=0.15,color='#C62828',ls='--',lw=0.6);a2.set_xlim(0,cd*1.03);a2.set_ylim(0,1.05)
    a2.set_xlabel('Distance [km]');a2.set_ylabel('SOC [–]');a2.set_title('(b) Battery depletion');cax(a2)
    fig.suptitle(f'{sc.sid}: {sc.name}',fontsize=11,y=1.01);plt.tight_layout()
    fig.savefig(f'{out}/fig05_multistop.png');plt.close(fig);print('  ✓ fig05')

def fig6(dem,air,aon,scs,ac,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for col,sid in enumerate(['S1','S3']):
        sc=[s for s in scs if s.sid==sid][0];lg0=aon[sid][0];ax=axes[col]
        wpr=lg0.rp.flight_path.get_waypoints_array();dk=wpr[:,4]/1000;terr=dem.elevation_batch(wpr[:,0],wpr[:,1]);v=~np.isnan(terr)
        ceil=terr.copy();ceil[v]+=120
        ax.fill_between(dk[v],0,terr[v],alpha=0.25,color='#8D6E63',lw=0)
        ax.fill_between(dk[v],terr[v],ceil[v],alpha=0.10,color='#43A047',lw=0)
        ax.fill_between(dk[v],ceil[v],ceil[v]+500,alpha=0.06,color='#EF5350',lw=0)
        ax.plot(dk[v],terr[v],'-',color='#5D4037',lw=0.6);ax.plot(dk[v],ceil[v],'--',color='#BF360C',lw=0.8,label='120m AGL ceiling')
        ax.text(dk[v][len(dk[v])//2],np.mean(terr[v])+60,'Flyable\ncorridor',fontsize=6,color='#2E7D32',ha='center',fontstyle='italic')
        wps=lg0.fp_straight.get_waypoints_array();ax.plot(wps[:,4]/1000,wps[:,2],'-',color=C_STR,lw=0.8,label='Straight',alpha=0.7)
        astar_profile(ax,lg0.astar_wps,dem)
        ax.plot(dk,wpr[:,2],'-',color=C_RDAC,lw=1.8,label='RDAC 101',path_effects=[pe.withStroke(linewidth=2.5,foreground='white')])
        ax.set_xlabel('Distance [km]')
        if col==0:ax.set_ylabel('Altitude [m]')
        lb='Terrain-dominated' if sid=='S1' else 'Zone-dominated'
        ax.set_title(f'({chr(97+col)}) {sc.sid} — {lb}');ax.legend(fontsize=5.5);cax(ax)
    fig.suptitle('Flyable Corridor: Terrain, AGL Ceiling, and Flight Paths',fontsize=11,y=1.01);plt.tight_layout()
    fig.savefig(f'{out}/fig06_corridor.png');plt.close(fig);print('  ✓ fig06')

def figS_conv(aon,scs,out):
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for idx,(sid,clr) in enumerate(zip(['S1','S3','S6','S8'],['#1565C0','#D32F2F','#2E7D32','#795548'])):
        ax=axes[idx//2,idx%2];sc=[s for s in scs if s.sid==sid][0];lg0=aon[sid][0]
        h=lg0.opt_result.convergence_history if hasattr(lg0.opt_result,'convergence_history') else []
        ph=lg0.opt_result.parameter_history if hasattr(lg0.opt_result,'parameter_history') else []
        gens=range(1,len(h)+1)
        if h:
            ax.plot(gens,h,'-',color=clr,lw=1.8,label='Objective')
            ax.set_xlabel('Generation');ax.set_ylabel('Objective',color=clr)
            ax.tick_params(axis='y',labelcolor=clr);cax(ax)
        ax.set_title(f'{sc.sid}: {sc.name}')
        # Overlay key design variables on twin axes
        if ph and len(ph)>1:
            ni=lg0.rp.n_intermediate;nl=ni+1
            # Extract variable traces from parameter snapshots
            p_arr=np.array(ph)  # shape: (n_snapshots, n_params)
            p_gens=np.linspace(1,len(h),len(ph))
            ax2=ax.twinx()
            if ni>0:
                # Lateral offset (index 0)
                lat_off=p_arr[:,0]/1000  # km
                ax2.plot(p_gens,lat_off,'--',color='#E65100',lw=1.0,alpha=0.8,label='Lat. offset [km]')
            # Mean cruise altitude (indices: ni to ni+nl)
            alt_cols=list(range(ni,ni+nl))
            if max(alt_cols)<p_arr.shape[1]:
                mean_alt=np.mean(p_arr[:,alt_cols],axis=1)
                ax2.plot(p_gens,mean_alt,'--',color='#388E3C',lw=1.0,alpha=0.8,label='Mean alt [m]')
            # Mean cruise speed (indices: ni+nl to ni+2*nl)
            spd_cols=list(range(ni+nl,ni+2*nl))
            if max(spd_cols)<p_arr.shape[1]:
                mean_spd=np.mean(p_arr[:,spd_cols],axis=1)
                ax2.plot(p_gens,mean_spd,':',color='#7B1FA2',lw=1.0,alpha=0.8,label='Mean V [m/s]')
            ax2.set_ylabel('Design variables',fontsize=7)
            ax2.tick_params(axis='y',labelsize=6)
            lines1,labels1=ax.get_legend_handles_labels()
            lines2,labels2=ax2.get_legend_handles_labels()
            ax.legend(lines1+lines2,labels1+labels2,fontsize=5.5,loc='center right')
    fig.suptitle('DE Convergence: Objective and Key Design Variables',fontsize=11,y=1.01);plt.tight_layout()
    fig.savefig(f'{out}/figS_convergence.png');plt.close(fig);print('  ✓ figS_conv')

def figS_evolution(dem,air,uav,con,ac,sc,MI,POP,SD,out):
    o,d,pay=sc.legs[0].origin,sc.legs[0].destination,sc.legs[0].payload_kg
    opt=PathOptimizer(dem,uav,con,ac)
    stages=sorted(set([1,2,3,5,max(MI//4,1),max(MI//2,1),MI]));stages=[s for s in stages if s<=MI]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5.5));cm_=plt.cm.viridis;nm=plt.Normalize(0,len(stages)-1)
    al,an=[o.lat,d.lat],[o.lon,d.lon];pdata=[]
    for i,mi in enumerate(stages):
        rp=RoutedPath(o,d,dem,uav,con,n_intermediate=1)
        opt.optimize_routed(rp,mode=OptMode.ENERGY,payload_kg=pay,airspace=air,maxiter=mi,popsize=max(POP,4),verbose=False,seed=SD)
        wp=rp.flight_path.get_waypoints_array();e=analyze_path_energy(rp.flight_path,ac).total_energy_wh
        al.extend(wp[:,0]);an.extend(wp[:,1]);pdata.append((mi,wp,e))
    m=0.015;lat_r=(min(al)-m,max(al)+m);lon_r=(min(an)-m,max(an)+m)
    draw_dem(a1,dem,lat_r,lon_r);draw_zones(a1,air,lat_r,lon_r);a1.plot([o.lon,d.lon],[o.lat,d.lat],':',color='0.5',lw=0.5,zorder=3)
    for i,(mi,wp,e) in enumerate(pdata):
        c=cm_(nm(i));alpha=0.3+0.7*(i/max(len(pdata)-1,1));lw=0.6+1.4*(i/max(len(pdata)-1,1))
        a1.plot(wp[:,1],wp[:,0],'-',color=c,lw=lw,alpha=alpha,zorder=4+i,label=f'iter {mi}' if i in [0,len(pdata)//2,len(pdata)-1] else None)
    _,wpf,_=pdata[-1];a1.plot(wpf[:,1],wpf[:,0],'-',color=C_RDAC,lw=2.5,zorder=10,path_effects=[pe.withStroke(linewidth=3.5,foreground='white')])
    endpts(a1,o,d);a1.set_xlim(*lon_r);a1.set_ylim(*lat_r);a1.set_xlabel('Lon [°]');a1.set_ylabel('Lat [°]');a1.set_title(f'(a) Path evolution');a1.legend(fontsize=5.5)
    a2.plot([p[0] for p in pdata],[p[2] for p in pdata],'-o',color=C_RDAC,lw=1.5,ms=4,mec='k',mew=0.3)
    a2.set_xlabel('DE iterations');a2.set_ylabel('Energy [Wh]');a2.set_title('(b) Energy convergence');cax(a2)
    fig.suptitle(f'Optimization Evolution — {sc.sid}: {sc.name}',fontsize=11,y=1.01);plt.tight_layout()
    fig.savefig(f'{out}/figS_evolution_{sc.sid}.png');plt.close(fig);print(f'  ✓ figS_evolution ({sc.sid})')

# ═══ MAIN ═══
def find_dem():
    for p in ['data/dmq_dem.npz','dmq_dem.npz','../data/dmq_dem.npz']:
        if os.path.exists(p):return p
    raise FileNotFoundError("dmq_dem.npz not found")

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--maxiter',type=int,default=15);pa.add_argument('--popsize',type=int,default=6)
    pa.add_argument('--seed',type=int,default=42);pa.add_argument('--n-wp',type=int,default=1)
    pa.add_argument('--out',type=str,default='paper_figures');pa.add_argument('--main-only',action='store_true')
    a=pa.parse_args();MI,POP,SD=a.maxiter,a.popsize,a.seed;os.makedirs(a.out,exist_ok=True);set_style()
    print(f"CONDOR v2 | MI={MI} POP={POP}");dem=DEMInterface(find_dem())
    uav=UAVConfig();con=MissionConstraints();ac=AircraftEnergyParams();opt=PathOptimizer(dem,uav,con,ac)
    air=build_airspace(dem=dem);bld=PathBuilder(dem,uav,con);ast=AStarGridPlanner(dem,ac,airspace=air,grid_resolution_m=500)
    scs=build_scenarios(dem)
    print("\nRunning...");t0=time.time();aoff,aon={},{}
    for sc in scs:
        print(f"  {sc.sid}: {sc.name}...",end=' ',flush=True)
        aoff[sc.sid]=run_sc(sc,dem,uav,con,ac,opt,bld,ast,None,MI,POP,SD,a.n_wp)
        aon[sc.sid]=run_sc(sc,dem,uav,con,ac,opt,bld,ast,air,MI,POP,SD,a.n_wp)
        eo=sum(l.e_res.total_energy_wh for l in aoff[sc.sid]);en=sum(l.e_res.total_energy_wh for l in aon[sc.sid])
        print(f"OFF={eo:.0f} ON={en:.0f} SOC={aon[sc.sid][-1].soc_end*100:.0f}%")
    print(f"Done {time.time()-t0:.0f}s\n")
    with open(f'{a.out}/table2.md','w') as f:
        f.write("|ID|Name|Legs|Urg|E_OFF|E_ON|ΔE%|SOC_OFF|SOC_ON|F|\n|---|---|---|---|---|---|---|---|---|---|\n")
        for sc in scs:
            eo=sum(l.e_res.total_energy_wh for l in aoff[sc.sid]);en=sum(l.e_res.total_energy_wh for l in aon[sc.sid])
            de=(en-eo)/max(eo,1)*100 if eo>0 else float('nan');so=aoff[sc.sid][-1].soc_end;sn=aon[sc.sid][-1].soc_end
            f.write(f"|{sc.sid}|{sc.name}|{len(sc.legs)}|{sc.urgency}|{eo:.0f}|{en:.0f}|{de:+.0f}%|{so*100:.0f}%|{sn*100:.0f}%|{'✓' if sn>0.15 else '✗'}|\n")
    print('  ✓ table2')
    print("\n=== MAIN ===")
    fig1(dem,air,scs,a.out)
    fig2(dem,air,bld,opt,ast,uav,con,ac,[s for s in scs if s.sid=='S3'][0],MI,POP,SD,a.out)
    fig3(aoff,aon,scs,a.out)
    fig4(dem,air,aon,scs,a.out)
    fig5(dem,air,aon,scs,a.out)
    fig6(dem,air,aon,scs,ac,a.out)
    if not a.main_only:
        print("\n=== SUPPLEMENTARY ===")
        for sid in ['S5','S2']:fig2(dem,air,bld,opt,ast,uav,con,ac,[s for s in scs if s.sid==sid][0],MI,POP,SD,a.out)
        # S7 multistop
        sc7=[s for s in scs if s.sid=='S7'][0];l7=aon['S7'];c7=['#1565C0','#E65100','#2E7D32']
        fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5));al,an=[],[]
        for lg in l7:wp=lg.rp.flight_path.get_waypoints_array();al.extend(wp[:,0]);an.extend(wp[:,1])
        m=0.015;lr_r=(min(al)-m,max(al)+m);lo_r=(min(an)-m,max(an)+m)
        draw_dem(a1,dem,lr_r,lo_r);draw_zones(a1,air,lr_r,lo_r)
        for i,lg in enumerate(l7):path_arrowed(a1,lg.rp.flight_path,c7[i%3],f'Leg {i+1}',lw=1.5)
        a1.set_xlim(*lo_r);a1.set_ylim(*lr_r);a1.set_xlabel('Lon');a1.set_ylabel('Lat');a1.set_title('(a) Corridors');a1.legend(fontsize=5.5)
        cd=0
        for i,lg in enumerate(l7):
            wp=lg.rp.flight_path.get_waypoints_array();ld=wp[-1,4]/1000
            a2.fill_between([cd,cd+ld],lg.soc_start,lg.soc_end,alpha=0.25,color=c7[i%3]);a2.plot([cd,cd+ld],[lg.soc_start,lg.soc_end],'-o',color=c7[i%3],lw=2,ms=4,mec='k',mew=0.3);cd+=ld
        a2.axhline(y=0.15,color='#C62828',ls='--',lw=0.6);a2.set_xlim(0,cd*1.03);a2.set_ylim(0,1.05)
        a2.set_xlabel('Distance [km]');a2.set_ylabel('SOC');a2.set_title('(b) Battery');cax(a2)
        fig.suptitle(f'S7: {sc7.name}',fontsize=11,y=1.01);plt.tight_layout();fig.savefig(f'{a.out}/figS4_S7.png');plt.close(fig);print('  ✓ figS4')
        figS_conv(aon,scs,a.out)
        figS_evolution(dem,air,uav,con,ac,[s for s in scs if s.sid=='S3'][0],MI,POP,SD,a.out)
    print(f"\n{'='*50}")
    for f in sorted(os.listdir(a.out)):print(f"  {f}: {os.path.getsize(f'{a.out}/{f}')//1024} KB")

if __name__=='__main__':main()