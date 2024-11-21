import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hplib.hplib_database as db
import hplib.hplib as hpl
import concurrent.futures
import time

np.random.seed(1072024)
np.seterr(divide = 'ignore') 
database = hpl.load_database()

class BESS:
    def __init__(self, init_soc = 0, cap = 1, C_rate = 1, eta = 0.95):
        self.soc = init_soc # SOC of the BESS
        self.size = cap # capacity of the BESS
        self.c_rate = C_rate # C rating of the BESS
        self.eta = 0.95 # efficiency of the BESS per charge or discharge

    def charge(self, P):    # an algorithm to find the charging power for the BESS

        if self.size > 0:
            p_bat = -1 * min(
                (self.size/4),
                ((1 - self.soc) * ((self.size) / self.eta)/4),
                (-1 * P/4))
            
            new_soc = self.soc - (p_bat / self.size) * self.eta # updates the SOC of the BESS
        else:
            new_soc = 0
            p_bat = 0
        return p_bat*4, new_soc, 0


    def discharge(self, P):     # an algorithm to find the discharging power for the BESS
        if self.size > 0:
            p_bat = min(
                    (self.size/4),
                    (self.soc * self.size * self.eta/4),
                    (P/4))
            new_soc = self.soc - p_bat / (self.size * self.eta) # updates the SOC of the BESS
            dod = self.soc - new_soc # Finds the depth of discharge of the BESS for this specific discharge
        else:
            new_soc = 0
            p_bat = 0
            dod = 0
        return p_bat*4, new_soc, dod


class TESS: # a class related to all the TESS operations
    def __init__(self, init_temp = 20, max_temp = 60, cap_in_kwh = 30, maxP = 4500):
        self.temp = init_temp    # temperature of the TESS
        self.max_temp = max_temp    # max temp allowed for the TESS
        self.kwh = cap_in_kwh   # This would be the capacity in 
        self.C = 1.163   # Wh per kgC 
        self.maxhpP = maxP  # The maximum power for the heat pump. Provided by hplib as 4500
        self.size = cap_in_kwh/((max_temp-init_temp)*(self.C/1000)) # since most of the caluclations depend on the mass of the water within the TESS, this converts the kWh size to mass of water

    def charge(self, heatpump, current_Ta, P_el):       # finds the charging power in both thermal and electric along with the new temperature for the TESS
        h = heatpump.simulate(t_in_primary=current_Ta, t_in_secondary=self.temp+0.5, t_amb=current_Ta, mode=1)
        cop = h['COP']
        maxAllowed = (self.max_temp-self.temp)*self.size*self.C*4/cop   # finds the amximum allowed thermal power divided by the COP
        P = min(P_el*1000, self.maxhpP, maxAllowed)*cop
        dT = P/(self.size*self.C*4)
        new_T = self.temp + dT
        p_tess = P/(cop*1000)
        return p_tess, -P/1000, new_T


    def discharge(self, P_thermal):     # finds the discharging power, and the new TESS temperature
        dT = P_thermal*1000/(self.size*self.C*4)
        new_T = self.temp - dT
        p_th = P_thermal
        return 0, p_th, new_T
    
    def chispossible(self):     # checks if charging the TESS is possible
        if self.temp < self.max_temp and self.size > 0:
            return True
        else:
            return False
    
    def dischispossible(self, temp, p_thermal):     # checks if discharging the TESS is possible.
        if self.size*self.C*(self.temp - temp) > p_thermal*(1000/4) and p_thermal > 0 and self.size > 0:
            return True
        else:
            return False


class HydroESS:
    def __init__(self, init_soc = 0, cap = 1, eta_fc = 0.50, eta_electrolyzer = 0.74):
        self.size = cap
        self.eta = np.sqrt(eta_fc*eta_electrolyzer)
        self.soc = init_soc
        self.ezcap = cap
        self.fccap = cap
        self.ezeta = eta_electrolyzer
        self.fceta = eta_fc
        self.sizekg = cap*4
        self.kwhperkg = 33.33
        self.kwminperkg = self.kwhperkg*4

    def charge(self, P):    # similar to BESS charge method, but for HESS
        if self.ezcap > 0:
            p_hyd = -1 * min(
                    (self.ezcap/4),
                    ((self.sizekg - self.soc) * (self.kwhperkg / self.ezeta)/4),
                    (-1 * P/4))
            
            new_soc = self.soc - (p_hyd * self.ezeta / self.kwhperkg)
        else:
            new_soc = 0
            p_hyd = 0
        return p_hyd*4, new_soc


    def discharge(self, P):     # similar to BESS discharge method, but for HESS
        if self.fccap > 0 :
            p_hyd = min(
                    (self.fccap/4),
                    (self.soc * self.kwhperkg * self.fceta/4),
                    (P))
            
            new_soc = self.soc - (p_hyd / (self.fceta * self.kwhperkg))
        else:
            new_soc = self.soc
            p_hyd = 0
        return p_hyd*4, new_soc


class Econ:     # Economic class
    def __init__(self, eco):
        self.ppv = eco['ppv'][0]
        self.ppe = eco['ppe'][0]
        self.pbos = eco['pbos'][0]
        self.fepc = eco['fepc'][0]
        self.fop = eco['fop'][0]
        self.pfi = eco['pfi'][0]
        self.pel = eco['pel'][0]
        self.iinv = eco['iinv'][0]
        self.ivat = eco['ivat'][0]
        self.tinv = eco['tinv'][0]
        self.lpv = eco['lpv'][0]
        self.lpe = eco['lpe'][0]
        self.pb = eco['pb'][0]
        self.lcycbat = eco['lcycbat'][0]
        self.lifbat = eco['lifbat'][0]
        self.pt = eco['pt'][0]
        self.phkg = eco['phkg'][0]
        self.pez = eco['pez'][0]
        self.pfc = eco['pfc'][0]
        self.lht = eco['lht'][0]
        self.lez = eco['lez'][0]
        self.lfc = eco['lfc'][0]
        self.php = eco['php'][0]
    
    def getNPV(self, bdf, cappv, BESS, TESS, HydroESS): #calculates the NPV
        #all the initial investments are here
        #Cpv = (1+self.ivat)*(self.ppv+self.ppe+self.pbos)*(1+((self.fepc**-1)-1)**-1)*cappv
        Cpv = (1+self.ivat)*(self.ppv)*cappv
        Cbess = (1+self.ivat)*self.pb*BESS.size
        Ctess = (1+self.ivat)*self.pt*TESS.kwh
        Cht = (1+self.ivat)*(self.phkg*HydroESS.sizekg)
        Cpez = (1+self.ivat)* (self.pez* HydroESS.ezcap)
        Cpfc = (1+self.ivat)* (self.pfc* HydroESS.fccap)
        Chydro = Cht + Cpez + Cpfc
        Cpe = (1+self.ivat)*self.ppe*BESS.size
        Chp = (1+self.ivat)*self.php
    
        NPV = Cpv + Cbess + Ctess + Chydro + Cpe + Chp
        NPVnosys = 0

        rg = 525600//15
        Cpvop = Cpv*self.fop
        bat_cycles = 0
        bat_yr = 0
        for i in range(1,21):
            fisum= bdf[bdf['P_d']<0]['P_d'][int((i-1)*rg):int((i)*(rg-1))].sum()        # gets the sum of all the fed in electricity
            elsum = bdf[bdf['P_d']>0]['P_d'][int((i-1)*rg):int((i)*(rg-1))].sum()       # gets the sum of all the electricity bought from the grid
            rev = (1+self.ivat)*self.pfi*(fisum/4)+self.pel*(elsum/4)                   # revenue

            hhsum = self.pel*bdf['P_tot'][int((i-1)*rg):int((i)*(rg-1))].sum()/4
            #NPVnosys += hhsum/((1+self.iinv)**i)
            NPVnosys += self.pel*(bdf['P_tot'][(35040)*(i-1):(35040-1)*i].sum()/4)/((1+0.02)**i)    # The NPV if there was no system installed. Obsolete, but I keep it in the code for more information when needed.

            NPV +=  (rev + Cpvop)/((1+self.iinv)**i)
            

            bat_yr += 1
            bat_cycles += bdf['dod'][int((i-1)*rg):int((i)*(rg-1))].sum()
            # below it checks if the lifetimes have been exceeded and the component needs to be replaced
            if (i-1) % self.lpv == 0 and i != 1:
                NPV += Cpv/((1+self.iinv)**i)
            if (i-1) % self.lpe == 0 and i != 1:
                NPV += Cpe/((1+self.iinv)**i)
            if (i-1) % self.lht == 0 and i != 1:
                NPV += Cht/((1+self.iinv)**i)
            if (i-1) % self.lez == 0 and i != 1:
                NPV += Cpez/((1+self.iinv)**i)
            if (i-1) % self.lfc == 0 and i != 1:
                NPV += Cpfc/((1+self.iinv)**i)
            if bat_cycles >= self.lcycbat:
                bat_cycles += -self.lcycbat
                bat_yr = 0
                NPV += Cbess/((1+self.iinv)**i)
            if bat_yr > self.lifbat:
                bat_cycles = 0
                bat_yr = 0
                NPV += Cbess/((1+self.iinv)**i)
        realNPV = (NPV - NPVnosys)
        return NPV




class PVspec:       # a class for the specs of the pv panels
    noct = 25
    anoct = 20
    n_ref = 0.21
    taoa = 0.9
    beta = 0.0048
    tcref = 25
    gamma = 0.12
    g_noct = 800
    g_ref = 1000

                    
def PVCalc(TMY):    # calcualtes the PV generation per kW
    AreaPerkW = 4.76
    T_c = TMY["Ta"] + (TMY["G_Gk"]/PVspec.g_noct) * (9.5/(5.7+3.8*TMY["FF"])) * (PVspec.noct-PVspec.anoct) * (1 - PVspec.n_ref/PVspec.taoa)
    n_pv = PVspec.n_ref * (1 - PVspec.beta*(T_c-PVspec.tcref) + 0.12*(np.log(TMY["G_Gk"]/PVspec.g_ref)))
    n_pv.replace(-np.inf, 0, inplace=True)
    PVGen = n_pv*TMY["G_Gk"]*AreaPerkW*0.95 # Total PV generation per kW. the 0.95 is the inverter's efficiency

    return PVGen/1000


def runSim(PV_size,BESS,TESS,HydroESS, pvgen,loadcons,TMY):     # the main method to simulate the PV + Hybrid energy storage system
  
    pv_gen_perA = np.array(pvgen)
    pv_gen_perA = np.tile(pv_gen_perA, 20)          # provide the generation for all the 20 years
    load_list = loadcons
    
    length = len(pv_gen_perA)

    # all the lists that would have the values that show up in the df
    soc_list = np.zeros(length)
    p_b_list = np.zeros(length)
    p_d_list = np.zeros(length)
    hh_temp = np.zeros(length)
    p_th = np.zeros(length)
    p_el = np.zeros(length)
    p_hyd_list = np.zeros(length)
    tess_temp = np.zeros(length)
    hyd_soc_list = np.zeros(length)
    dod_list = np.zeros(length)
    p_tot_list = np.zeros(length)

    current_hh_temp = 21    # an assumption that all households start with 21 degrees as their initial temperature

    parameters = hpl.get_parameters('i-SHWAK V4 06')        # loads the parameters of the 
    heatpump=hpl.HeatPump(parameters)
    
    
    TaVals = TMY['Ta'].values
    TaVals = np.tile(TaVals, 20)
    
    # all the heat model calculations are below.
    no_of_hh = 5
    height = 3
    U_roof = 0.2
    A_roof = 93
    U_window = 1.3
    A_window = 4
    U_wall = 0.28
    A_wall = 9.64*height*4 - A_window

    volume = A_roof*height * no_of_hh
    c_air = 1005
    pho_air = 1.225
    m_air = volume * pho_air

    Q_roof_C = U_roof*A_roof
    Q_window_C = U_window*A_window
    Q_wall_C = U_wall*A_wall

    Q_C = (Q_roof_C + Q_window_C + Q_wall_C)*no_of_hh

    pv_gen_perA = pv_gen_perA* PV_size
    Q_loss_constant = (m_air*c_air)/900     # divided by 900 because it is 15 mins step.

    dod = 0
    

    for i in range(length):

        current_pv_gen = pv_gen_perA[i]
        current_load = load_list[i]
        current_Ta = TaVals[i]

        Q_loss = Q_C * (current_hh_temp - current_Ta)
        delta_T = (Q_loss/Q_loss_constant)
        new_T = TESS.temp
        p_ther = 0
        p_hp = 0
        dod = 0


        current_hh_temp = current_hh_temp - delta_T

        # calculates the thermal requirements depending on the temperature
        if current_hh_temp < 18 :
            COP = heatpump.simulate(t_in_primary=current_Ta, t_in_secondary=21.5, t_amb=current_Ta, mode=1)['COP'] 
            current_thermalLoad = Q_loss_constant * (25 - current_hh_temp)/(1000*2)
            current_hp_load = current_thermalLoad/COP
            new_hh_temp = current_hh_temp + (current_thermalLoad*1000/(Q_loss_constant))
            
            

        elif current_hh_temp > 25 :
            COP = heatpump.simulate(t_in_primary=current_Ta, t_in_secondary=21.5, t_amb=current_Ta, mode=2)['EER']
            current_thermalLoad = Q_loss_constant * (20 - current_hh_temp)/(1000*2)
            current_hp_load = -current_thermalLoad/COP
            new_hh_temp = current_hh_temp + (current_thermalLoad*1000/(Q_loss_constant))

        else:
            new_hh_temp = current_hh_temp 
            current_hp_load = 0
            current_thermalLoad = 0
            
        
        p_required = current_load  - current_pv_gen
        
        
        
        if p_required <= 0: # overgeneration -  charge

                p_required2 = p_required + current_hp_load
                if p_required2 < 0: # PV covers hp load
                    p_bat, new_soc, dod = BESS.charge(p_required2)
                    
                    p_d = p_required2 - p_bat

                    if p_d < 0 and TESS.chispossible():
                        p_hp, p_ther, new_T = TESS.charge(heatpump,current_Ta,-(p_required2-p_bat))
                        p_d = p_d + p_hp

                    if p_d < 0:
                        p_hyd, hyd_new_soc = HydroESS.charge(p_d)
                        p_d = p_d - p_hyd


                elif p_required2 > 0: # pv cannot cover hp load, go with TESS discharge
                    if TESS.dischispossible(current_hh_temp, current_thermalLoad):
                        p_hp, p_ther, new_T = TESS.discharge(current_thermalLoad)

                        p_d = p_required2 - current_hp_load 
                        

                    elif BESS.soc > 0 : # if TESS was not discharged, use BESS to power up the heat pump
                        p_bat, new_soc, dod = BESS.discharge(p_required2)
                        p_d = p_required2 - p_bat

                        if p_bat < current_hp_load:
                            p_hyd, hyd_new_soc = HydroESS.discharge(current_hp_load - p_bat)
                            p_d = p_required2 - p_bat - p_hyd                            
                    
                    elif HydroESS.soc > 0: # If BESS not available, then HESS to power up hp
                        p_hyd, hyd_new_soc = HydroESS.discharge(current_hp_load)
                        p_d = p_required2 - p_hyd 

                    else:
                        p_d = p_required2

        elif p_required > 0: # deficit in electricity
                if TESS.dischispossible(current_hh_temp, current_thermalLoad):
                    p_hp, p_ther, new_T = TESS.discharge(current_thermalLoad)
                    p_required2 = p_required 
                else:
                    p_required2 = p_required + current_hp_load


                p_bat, new_soc, dod = BESS.discharge(p_required2)

                p_d = p_required2 - p_bat

                p_hyd, hyd_new_soc = HydroESS.discharge(p_d)
                p_d = p_d - p_hyd

                
        

        p_d_list[i] = p_d
        soc_list[i] = new_soc
        dod_list[i] = dod
        hyd_soc_list[i] = hyd_new_soc
        tess_temp[i] = new_T
        p_b_list[i] = p_bat
        p_hyd_list[i] = p_hyd
        p_th[i] = p_ther
        p_el[i] = p_hp
        hh_temp[i] = new_hh_temp
        p_tot_list[i] = current_load - current_pv_gen + current_hp_load
        


        current_hh_temp = new_hh_temp
        BESS.soc = new_soc
        HydroESS.soc = hyd_new_soc
        TESS.temp = new_T
        

    Simdf = pd.DataFrame({
        'soc': soc_list,
        'dod': dod_list,
        'P_b': p_b_list,
        'P_HH': load_list,
        'P_tot': p_tot_list,
        'P_d': p_d_list,
        'hh_temp': hh_temp,
        'Ta': TaVals,
        'P_th': p_th,
        'P_el': p_el,
        'tess_temp': tess_temp,
        'P_hyd': p_hyd_list,
        'hyd_soc': hyd_soc_list
    })


    return Simdf


#class Optimize:
    def __init__(self, pv_gen_perA,load_list,TMY, ecoData):
        self.pv_gen_perA = pv_gen_perA
        self.load_list = load_list
        self.TMY = TMY
        self.ecoData = ecoData

    def PSO(self, pop = 3, iterations = 10, c1 = 1, c2 = 1, w_max = 0.9, w_min = 0.4, lower_bound = 0, upper_bound = 10):
        pos = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop, 4))
        velocity = np.zeros_like(pos)
        personal_best_pos = pos.copy()
        personal_best_val = np.full(pop, np.inf)
        gloabl_best_pos = np.zeros(4)
        global_best_val = np.inf

        W = np.linspace(w_max, w_min, iterations)
        eco =Econ(self.ecoData)

        for c in range(iterations): 
            w = W[c]
            for i in range(pop):
                r1 = np.random.rand()
                r2 = np.random.rand()
                velocity[i] = (w * velocity[i] + c1 * r1 * (personal_best_pos[i] - pos[i]) + c2 * r2 * (gloabl_best_pos- pos[i]))
                
                pos[i] = pos[i] + velocity[i]  + np.random.uniform(-0.25, 0.25, size=4)
                pos[i] = np.clip(pos[i], lower_bound, upper_bound)


                p = pos[i]
                cappv = p[0]
                b = BESS(cap = p[1])
                t = TESS(cap_in_kwh=p[2])
                h = HydroESS(cap = p[3])

                bat = runSim(cappv,b,t,h,gen,loadData, TMYData)

                eco = Econ(ecoData)

                NPV = eco.getNPV(bat, cappv,b,t,h)
                print(NPV)


                if NPV < personal_best_val[i]:
                    personal_best_val[i] = NPV
                    personal_best_pos[i] = pos[i]
                    print(f"NEW personal for {i}! pos: {personal_best_pos[i]}, NPV: {personal_best_val[i]}")

                if NPV < global_best_val:
                    global_best_val = NPV
                    gloabl_best_pos = pos[i]
                    print(f"NEW global! pos: {gloabl_best_pos}, NPV: {global_best_val}")
                

        return gloabl_best_pos, global_best_val, personal_best_pos, personal_best_val


def getLoadData(ecoData):       # gets the load data for the 20 years.
    e = Econ(ecoData)
    rand1 = np.random.randint(1,101, size = e.tinv)
    rand2 = np.random.randint(101,201, size = e.tinv)
    rand3 = np.random.randint(201,301, size = e.tinv)
    rand4 = np.random.randint(301,401, size = e.tinv)
    rand5 = np.random.randint(401,501, size = e.tinv)
    rand = np.array([rand1, rand2, rand3, rand4, rand5])
    hh = np.zeros(10512000//15)

    for i in range(len(rand)):
        d = np.array([])
        for j in rand[i]:
            data = pd.read_csv(f"eni_seminar/Data/hh_{i+1}/H{j}.csv", header = None)[0].values

            start_datetime = '2001-01-01 00:00:00' # 2001 because i dont want a leap year, I do not think it matters anyway
            df = pd.DataFrame(data)
            datetime_range = pd.date_range(start=start_datetime, periods=len(df), freq='T')
            df['datetime'] = datetime_range
            df.set_index('datetime', inplace=True)
            df_resampled = df.resample('15T').mean()
            df_resampled.reset_index(inplace=True)
            data = df_resampled[0].values

            d = np.append(d, data)
        print(i)
        hh += d


    return hh
        
   
    # a function to evaluate the particle's NPV and position
def evaluate_particle(particle, gen, loadData, TMYData, ecoData):
    cappv, bess_cap, tess_cap, hydroess_cap = particle

    b = BESS(cap=bess_cap)
    t = TESS(cap_in_kwh=tess_cap)
    h = HydroESS(cap=hydroess_cap)

    sim = runSim(cappv, b, t, h, gen, loadData, TMYData)

    eco = Econ(ecoData)
    NPV = eco.getNPV(sim, cappv, b, t, h)
    
    return NPV




def main():


    TMYData = pd.read_csv("eni_seminar/Data/Munich_weather.csv")

    start_datetime = '2024-01-01 00:00:00'
    datetime_range = pd.date_range(start=start_datetime, periods=len(TMYData), freq='T')
    TMYData['datetime'] = datetime_range
    TMYData.set_index('datetime', inplace=True)
    TMYData_resampled = TMYData.resample('15T').mean()
    TMYData_resampled.reset_index(inplace=True)
    TMYData = TMYData_resampled

    ecoData = pd.read_csv("eni_seminar/Data/econ_para.csv")
    gen = PVCalc(TMYData_resampled)
    gen[gen<0] = 0
    loadData = getLoadData(ecoData) 

    # just for testing
    '''NPVv = 0
    for i in range(1,21):
        NPVv +=  0.4162*(loadData[(35040)*(i-1):(35040-1)*i].sum()/4)/((1+0.02)**i)
    print(f"NPV for 0 is: {NPVv}")
'''




    # Also for testing
    '''p = [0, 0, 0, 0]
    cappv = p[0]
    b = BESS(cap = p[1])
    t = TESS(cap_in_kwh=p[2])
    h = HydroESS(cap = p[3])

    print(f"{cappv} and {b.size} and {t.size} and {h.size}")

    bat = runSim(cappv,b,t,h,gen,loadData, TMYData)

    eco = Econ(ecoData)

    NPV = eco.getNPV(bat, cappv,b,t,h)
    print(NPV)'''

    lower_bound = 0
    upper_bound = 1000
    pop = 6
    iterations = 40
    w_max =0.9
    w_min = 0.4
    c1 = 1
    c2 = 0.5

    #pos = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop, 4))
    pos = np.random.random((pop,4))* [48.12566095, 19.79986607, 74.97941448,  0.319453  ] # This was the previously found optimal solution, I am using it to provide a guide to find a similar optimal soluiton. Otherwise it would need to take more iterations, and that would be computationally expensive.
    velocity = np.zeros_like(pos)
    personal_best_pos = pos.copy()
    personal_best_val = np.full(pop, np.inf)
    gloabl_best_pos = pos[0].copy()
    global_best_val = np.inf
    w = 0.9
    W = np.linspace(w_max, w_min, iterations)
    results = []

    for z in range(iterations): 
        w = W[z]
        start = time.perf_counter()
        for q in range(pop):
            r1 = np.random.rand()
            r2 = np.random.rand()
            velocity[q] = (w * velocity[q] + c1 * r1 * (personal_best_pos[q] - pos[q]) + c2 * r2 * (gloabl_best_pos- pos[q]))   # calculates the velocity. Check report for detailed undestanding
            
            pos[q] = pos[q] + velocity[q]  + np.random.uniform(0, 0.25, size=4) # updates position, and there is a small random number added, to avoid the particle being stuck in a local minimum (mainly 0,0,0,0)
            pos[q] = np.clip(pos[q], lower_bound, upper_bound)

        # you may consider changing this snippet if code to a loop if you do not want the program to run on more than one CPU.
        # processes the simulation concurrently using more resources
        with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(evaluate_particle, pos, [gen]*pop, [loadData]*pop, [TMYData_resampled]*pop, [ecoData]*pop))

        npvs = list(results)

        print(npvs)

        for q in range(len(npvs)):
            if npvs[q] < personal_best_val[q]:
                personal_best_val[q] = npvs[q]
                personal_best_pos[q] = pos[q]
                print(f"NEW personal for {q}! pos: {personal_best_pos[q]}, NPV: {personal_best_val[q]}")

            if npvs[q] < global_best_val:
                global_best_val = npvs[q]
                gloabl_best_pos = pos[q]
                print(f"NEW global! pos: {gloabl_best_pos}, NPV: {global_best_val}")

        finish = time.perf_counter()
        print(pos)
        print(f"iteration: {z}")
        print(finish-start)

    print(f"pos: {gloabl_best_pos}, NPV: {global_best_val}")

if __name__ == '__main__':
    main()