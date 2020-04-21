import pylab as pl
import numpy as np
import matplotlib.colors as colors
import glob,os,sys
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import column, row
from bokeh.plotting import figure
from datetime import *
try:
    from ipywidgets import *
except:
    from IPython.html.widgets import *
import subprocess
from IPython.display import display,clear_output
pl.matplotlib.rcParams.update({'font.size': 12})
class plot_gauss:

    def __init__(self,imode):
        self.Q = 200                     # Source strength [kg/s]
        self.U = 5                       # Mean wind [m/s]
        self.source_type = 'point'       # 'point', 'reflection', 'line', line has reflection
        self.H = 50                     # Height of source [m]
        self.xs = 0                      # x position of the source
        self.ys = 0                      # y position of the source
        #self.units = 1e9                 # Units of concentration, currently in micrograms for kg set to 1

        self.stab = 'stable'           # Stability: 'stable', 'neutral', 'unstable'
        self.transect = 'y-z'            # Which cross-section: 'y-z', 'x-z', 'x-y'
        self.distance = 3000             # Distance from origin where to take chosen cross-section [m]
        self.X = 7200
        self.Y = 5200
        self.Z = 3200
        self.dx = 10
        self.dy = 10
        self.dz = 10
        self.x = np.linspace(0.1, self.X, int(self.X/self.dx))
        self.y = np.linspace(-np.floor(self.Y/2), np.floor(self.Y/2), int(self.Y/self.dy))
        self.z = np.linspace(0, self.Z, int(self.Z/self.dz))
        self.type = Dropdown(options=['point','point+reflection','line+reflection'],value='point',description='Plume type')
        self.dist = IntSlider(description=':',value=3000,min=0,max=self.X-self.dx)
        self.sheight = IntSlider(description=':',value=50,min=0,max=1000)
        self.wind    = IntSlider(description=':',value=5,min=1,max=20)
        self.sstrength = IntSlider(description=':',value=200,min=1,max=1000)
        self.trans   = Dropdown(options=['x-y','x-z','y-z'],value='y-z',description='Transect:')
        self.stability = Dropdown(options=['stable','neutral','unstable'],value='unstable',description='Stability:')
        self.units = Dropdown(options=['kg/m3','ug/m3'],value='ug/m3',description='Unit:')
        self.wdoit = ToggleButton(description='Calculate',value=False)
        self.output = Checkbox( value=False , description='1D:')
        form_item_layout = Layout(
            display='flex',
            flex_flow='row',
            justify_content='flex-end'
            )
        self.unit0 = Label(value='kg/s ',layout=Layout(width='120px'))
        self.unit1 = Label(value='m ',layout=Layout(width='120px'))
        self.unit2 = Label(value='m/s ',layout=Layout(width='120px'))
        if self.type.value == 'line+reflection':
            self.unit0.value = 'kg/(ms)'
        else:
            self.unit0.value = 'kg/s'
        form_items = [
            Box([Label(value='Q : Strength of the source'),self.sstrength,self.unit0 ], layout = form_item_layout),
            Box([Label(value='H : Height of the source ') ,self.sheight,self.unit1], layout = form_item_layout),
            Box([Label(value='u : Wind speed at source ') ,self.wind,   self.unit2], layout = form_item_layout),
            Box([Label(value= ' '), self.output, Label(value=' ')], layout = form_item_layout)
            ]
        self.form = Box(form_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='70%'
            ))
        self.np   = BoundedIntText( value=1000, min=100, max=3000, step=100, description='npoints:')
        self.xpos = BoundedIntText( value=3000, min=1, max=11000, step=1, description='x-position:')
        self.ypos = BoundedIntText( value=0, min=-2700, max=2700, step=1, description='y-position:')
        self.zpos = BoundedIntText( value=50, min=0, max=3200, step=1, description='z-position:')
        self.dxpos = BoundedIntText( value=10, min=1, max=100, step=1, description='dx:')
        self.dypos = BoundedIntText( value=10, min=1, max=100, step=1, description='dy:')
        self.dzpos = BoundedIntText( value=10, min=1, max=100, step=1, description='dz:')
        self.conc  = FloatText( value=0.0, description='C (ug/m3):')
        if imode == 2: self.type.value='point+reflection'
        
        form_items2 = [
            Box([self.type,self.np ], layout = form_item_layout),
            Box([self.xpos,self.dxpos ], layout = form_item_layout),
            Box([self.ypos,self.dypos ], layout = form_item_layout),
            Box([self.zpos,self.dzpos ], layout = form_item_layout),
            Box([self.stability,self.conc ], layout = form_item_layout)
            ]
        self.form2 = Box(form_items2, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='70%'
            ))
        self.set_visibility(imode)
        output_notebook()
        display(self.form,self.form2)
        wmain = interact(self.xplot,doit = self.wdoit)

    def xplot(self,doit = ToggleButton()):
        #print('Source Strength (kg/s)',self.Q)
        #print('Wind Speed (m/s)',self.wind.value)
        #print('Height Source (m)',self.sheight.value)
        #print('x Distance (m)',self.dist.value)
        # calculate grid:
        if self.type.value == 'line+reflection':
            self.unit0.value = 'kg/(ms)'
        else:
            self.unit0.value = 'kg/s'
        n = self.np.value
        dx = self.dxpos.value
        dy = self.dypos.value
        dz = self.dzpos.value
        xp = self.xpos.value
        yp = self.ypos.value
        zp = self.zpos.value
        xmin = xp-n*dx/2
        xmax = xp+n*dx/2
        ymin = yp-n*dy/2
        ymax = yp+n*dy/2
        zmin = zp-n*dz/2
        zmax = zp+n*dz/2
        x = np.linspace(xmin,xmax, 1+int((xmax-xmin)/dx))
        self.y = np.linspace(ymin,ymax, 1+int((ymax-ymin)/dy))
        z = np.linspace(zmin,zmax, 1+int((zmax-zmin)/dz))
        self.x = x[np.where(x>0)]
        self.z = z[np.where(z>=0)]
        self.ixpos = np.where(self.x==xp)[0][0]
        self.iypos = np.where(self.y==yp)[0][0]
        self.izpos = np.where(self.z==zp)[0][0]
        #print(xmin,xmax,self.x)
        #print(ymin,ymax,self.y)
        #print(zmin,zmax,self.z)
        yz,xz,xy = self.gauss()
        if self.output.value:
            self.plot_gauss1D(yz,xz,xy)
        else:
            self.plot_gauss(yz,xz,xy)
        #self.wdoit.value=False
        return

    def gauss(self):
    
        # Dispersion parameters
    
        if self.stability.value == 'stable':
            F, f, G, g = 0.31, 0.71, 0.06, 0.71
            
        elif self.stability.value == 'neutral':
            F, f, G, g = 0.32, 0.78, 0.22, 0.78
            
        else:
            F, f, G, g = 0.40, 0.91, 0.41, 0.91
        
       # sigma_y = F*self.x**f
       # sigma_z = G*self.x**g

        #dist_ind = int(float(dist)/self.dx)
    
        #print("At %.i m" %(self.dist.value), "Dispersion parameter y: %.f m" %(sigma_y[dist_ind]), " Dispersion parameter z: %.f m" %(sigma_z[dist_ind]))
        stype = self.type.value
        if stype == 'point':
            s = 0
        else:
            s = 1
        if stype == 'line+reflection':
            l = 0
        else:
            l = 1
        if self.units.value == 'ug/m3':
            scale =1e9 
        else:
            scale = 1.0
#       yz cross-section at x
        U = self.wind.value 
        H = self.sheight.value
        sigma_y = F*self.xpos.value**f
        sigma_z = G*self.xpos.value**g
        if stype == 'line+reflection':
            print("At x = %.i m" %(self.xpos.value), "Dispersion parameter z: %.f m" %(sigma_z))
        else:
            print("At x = %.i m" %(self.xpos.value), "Dispersion parameter y: %.f m" %(sigma_y), " Dispersion parameter z: %.f m" %(sigma_z))

        if stype == 'line+reflection':
            norm =  1e9*self.sstrength.value/ (np.sqrt(2*np.pi)*sigma_z*U)
        else:
            norm =  1e9*self.sstrength.value / (2*np.pi*sigma_y*sigma_z*U)
        plume = norm * np.exp(- l*(self.ypos.value- self.ys )**2 / (2*sigma_y**2) ) *\
            (    np.exp(- (self.zpos.value - H )**2 / (2*sigma_z**2) ) + \
             s * np.exp(- (self.zpos.value + H )**2 / (2*sigma_z**2) ) )
        self.conc.value = plume
        plume_yz = norm * np.exp(- l*(self.y[None,:,None]- self.ys )**2 / (2*sigma_y**2) ) *\
            (    np.exp(- (self.z[None,None,:] - H )**2 / (2*sigma_z**2) ) + \
             s * np.exp(- (self.z[None,None,:] + H )**2 / (2*sigma_z**2) ) )
        # make sigmas x-dependent:  
        sigma_y = F*self.x**f
        sigma_z = G*self.x**g
        if stype == 'line+reflection':
            norm =  1e9*self.sstrength.value/ (np.sqrt(2*np.pi)*sigma_z[:,None,None]*U)
        else:
            norm =  1e9*self.sstrength.value / (2*np.pi*sigma_y[:,None,None]*sigma_z[:,None,None]*U)
        plume_xz = norm * np.exp(-l*(self.ypos.value-self.ys )**2 / (2*sigma_y[:,None,None]**2) ) *\
                        ( np.exp(-  (self.z[None,None,:] - H )**2 / (2*sigma_z[:,None,None]**2) ) + \
                      s * np.exp(-  (self.z[None,None,:] + H )**2 / (2*sigma_z[:,None,None]**2) ) )

        if stype == 'line+reflection':
            norm =  1e9*self.sstrength.value/ (np.sqrt(2*np.pi)*sigma_z[:,None,None]*U)
        else:
            norm =  1e9*self.sstrength.value / (2*np.pi*sigma_y[:,None,None]*sigma_z[:,None,None]*U)
        plume_xy = norm * np.exp(- l*(self.y[None,:,None]- self.ys )**2 / (2*sigma_y[:,None,None]**2) ) *\
                (   np.exp(- (self.zpos.value - H )**2 / (2*sigma_z[:,None,None]**2) ) + \
                s * np.exp(- (self.zpos.value + H )**2 / (2*sigma_z[:,None,None]**2) ) )
        return plume_yz,plume_xz,plume_xy

    def plot_gauss(self, yz,xz,xy):
        ## Always plots through the centerline
        
        fig, axsn = pl.subplots(1, 3, figsize=(15,6))
        if self.units.value == 'ug/m3':
            scale = 1e9
        else:
            scale = 1.0

        
        A, B = self.y*1e-3, self.z*1e-3
        yz=yz[0] 
        if self.type.value == 'line+reflection':
            ssum = yz[0,1:-2].sum() + 0.5*yz[0,0] + 0.5*yz[0,-1]
            print('Integral c*u*dz %8.1f kg/(m.s) '%(ssum*self.dzpos.value*self.wind.value/scale))
        else:
            ssum = yz[1:-2,1:-2].sum()+0.5*(yz[0,:].sum() + yz[0,-1].sum() + yz[:,0].sum() + yz[:,-1].sum())   # now we only count the corners double!
            print('Integral c*u*dz*dy %8.1f kg/s '%(ssum*self.dypos.value*self.dzpos.value*self.wind.value/scale))
        if yz.max() > 1e-2:
            axs = axsn[0]
            im = axs.pcolormesh(A, B, np.transpose(yz), norm=colors.LogNorm(vmin= 1e-2, vmax=yz.max() ))
            axs.set_title("yz plane at location x = %.i m"%(self.xpos.value))
            cb = fig.colorbar(im, ax = axs, orientation='horizontal')
            cb.ax.tick_params(labelsize=12) 
            cb.ax.set_xlabel('C [ug/m3]', fontsize=12)
            axs.tick_params(axis='both', which='major', labelsize=12)
            
            axs.set_ylabel('z [km]', fontsize = 12)
            axs.set_xlabel('y [km]', fontsize = 12)

        A, B = self.x*1e-3, self.z*1e-3
        axs = axsn[1]
        xz = xz[:,0,:]
        if xz.max() > 1e-2:
            im = axs.pcolormesh(A, B, np.transpose(xz), norm=colors.LogNorm(vmin= 1e-2, vmax=xz.max() ))
            axs.set_title("xz plane at location y = %.i m"%(self.ypos.value))
                
            cb = fig.colorbar(im, ax = axs, orientation='horizontal')
            cb.ax.tick_params(labelsize=12) 
            cb.ax.set_xlabel('C [ug/m3]', fontsize=12)
            axs.tick_params(axis='both', which='major', labelsize=12)
            
            axs.set_ylabel('z [km]', fontsize = 12)
            axs.set_xlabel('x [km]', fontsize = 12)

        A, B = self.x*1e-3, self.y*1e-3
        axs = axsn[2]
        xy = xy[:,:,0]
        if xy.max() > 1e-2:
            im = axs.pcolormesh(A, B, np.transpose(xy), norm=colors.LogNorm(vmin= 1e-2, vmax=xy.max() ))
            axs.set_title("xy plane at location z = %.i m"%(self.zpos.value))
                
            cb = fig.colorbar(im, ax = axs, orientation='horizontal')
            cb.ax.tick_params(labelsize=12) 
            cb.ax.set_xlabel('C [ug/m3]', fontsize=12)
            axs.tick_params(axis='both', which='major', labelsize=12)
            
            axs.set_ylabel('y [km]', fontsize = 12)
            axs.set_xlabel('x [km]', fontsize = 12)
            
        #elif transect == 'x-z':
        #    axs.set_ylabel('z [m]', fontsize = 12)
        #    axs.set_xlabel('x [m]', fontsize = 12)

        #else:
        #    axs.set_ylabel('y [m]', fontsize = 12)
        #    axs.set_xlabel('x [m]', fontsize = 12)
            
        pl.show(fig)

    def plot_gauss1D(self, yz,xz,xy):
        ### plot the values as 1D plots. For yz use plot from surface to zmax, for xz from x--> xmax, and for xy plot from ymin-ymax 
        # setup plot...
        pz = figure(title="at x,y = %.i m, %.i m"%(self.xpos.value,self.ypos.value), plot_height=300, plot_width=300)
        py = figure(title="at y,z = %.i m, %.i m"%(self.ypos.value,self.zpos.value), plot_height=300, plot_width=300)
        px = figure(title="at x,z = %.i m, %.i m"%(self.xpos.value,self.zpos.value), plot_height=300, plot_width=300)
        if self.units.value == 'ug/m3':
            scale = 1e9
        else:
            scale = 1.0
        A, B = self.y*1e-3, self.z*1e-3
        yz=yz[0] 
        if self.type.value == 'line+reflection':
            ssum = yz[0,1:-2].sum() + 0.5*yz[0,0] + 0.5*yz[0,-1]
            print('Integral c*u*dz %8.1f kg/(m.s) '%(ssum*self.dzpos.value*self.wind.value/scale))
        else:
            ssum = yz[1:-2,1:-2].sum()+0.5*(yz[0,:].sum() + yz[0,-1].sum() + yz[:,0].sum() + yz[:,-1].sum())   # now we only count the corners double!
            print('Integral c*u*dz*dy %8.1f kg/s '%(ssum*self.dypos.value*self.dzpos.value*self.wind.value/scale))
        if yz.max() > 1e-2:
            B = self.z*1e-3
            pz.line(yz[self.iypos,:],B,line_width = 2)
            pz.yaxis.axis_label = 'z [km]'
            pz.xaxis.axis_label = 'C [ug/m3]'
            flegend = 'x,y = %.i m, %.i m'%(self.xpos.value,self.ypos.value)
        A = self.x*1e-3
        xz = xz[:,0,:]
        if xz.max() > 1e-2:
            flegend = 'y,z = %.i m, %.i m'%(self.ypos.value,self.zpos.value)
            px.line(A, xz[:,self.izpos],line_width = 2)
            px.xaxis.axis_label = 'x [km]'
            px.yaxis.axis_label = 'C [ug/m3]'
        A = self.y*1e-3
        xy = xy[:,:,0]
        if xy.max() > 1e-2:
            flegend = 'x,z = %.i m, %.i m'%(self.xpos.value,self.zpos.value)
            py.line(A, xy[self.ixpos,:],line_width = 2)
            py.xaxis.axis_label = 'y [km]'
            py.yaxis.axis_label = 'C [ug/m3]'
        show(row(pz,px,py), notebook_hanle=True)
        push_notebook()


    def set_visibility(self,imode):
        if imode == 1:
            self.np.layout.visibility='hidden'
            self.type.layout.visibility='hidden'
            self.dxpos.layout.visibility='hidden'
            self.dypos.layout.visibility='hidden'
            self.dzpos.layout.visibility='hidden'
        elif imode == 2:
            self.np.layout.visibility='hidden'
            self.dxpos.layout.visibility='hidden'
            self.dypos.layout.visibility='hidden'
            self.dzpos.layout.visibility='hidden'
        elif imode ==3:
            self.xpos.value = 1000
            self.ypos.value = 0
            self.zpos.value = 2
            self.dxpos.value = 2
            self.dypos.value = 2
            self.dzpos.value = 2
            self.type.value = 'line+reflection'
            self.sheight.value = 5
            self.sstrength.value = 1


            

   

    
