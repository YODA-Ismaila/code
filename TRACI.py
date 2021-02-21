import os 
import sys 
import optparse 

from sumolib import checkBinary  
import traci 

path="C:/Users/yodai/Desktop/S7/Projet scientifique/code_prediction" 
os.chdir(path) 

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options
t=traci._trafficlight.TrafficLightDomain() #spécification du domaine traffic lights que nous utiliserons par la suite
t2=traci._lane.LaneDomain() #spécification du domaine:lanes(routes) que nous utiliserons par la suite
TLS=['gneJ1', 'gneJ2','gneJ9','gneJ10']

def run():
    step = 0  #initiation
    while step < 100: #Réglage du temps de simulation ici: 100 secondes
        traci.simulationStep()  #Lancement de la simulation sous SUMO
        if step==0: #Au début de la simulation
            LPI=[list(dict.fromkeys(list(t.getControlledLanes(tlsID=str(i))))) for i in TLS]#génération d'une liste qui comporte les lanes per intersection, càd les routes controllées par chacun des feux de circulation (l'intégration de la  méthode 'dict.fromkeys' a pour but d'éliminer les doublons dans les listes générées)
            LPI=sum(LPI,[]) #transformation en une liste à une dimension
            print(LPI,len(LPI))#Visualisation de la liste générée (peut être éliminée)
        if step%60==0: #Horizon de la prédiction =60secondes
            
            PREDICTION=resultat_prediction #exécution de l'opération de prédiction avec la fonction PRED prédéfinie
            print(PREDICTION) #Visualisation des prédictions générées
            CONTROL=resultat_control #exécution de l'opération du contrôle avec la fonction prédéfinie CONTR
            print(CONTROL) #Visualisation des résultats du contrôle
            t.setPhaseDuration(tlsID='gneJ1',phaseDuration=CONTROL[0]) #Changement de la phase du premier feu de circulation
            t.setPhaseDuration(tlsID='gneJ2',phaseDuration=CONTROL[1])#Changement de la phase du deuxième feu de circulation
            t.setPhaseDuration(tlsID='gneJ9',phaseDuration=CONTROL[3])#Changement de la phase du troisième feu de circulation
            t.setPhaseDuration(tlsID='gneJ10',phaseDuration=CONTROL[4]
            AFTER=[[t2.getWaitingTime(laneID=str(i))] for i in LPI] #Génération des temps d'attente moyens au niveau de chaque lane
            print("AFTER=%s"%(AFTER)) #Affichage de ces résultats
        step += 1 #incrémentation de la boucle
    traci.close() # Arrêt de la simulation
    sys.stdout.flush() #assurer la visualisation des outputs au moment de l'exécution du code

if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "config.sumocfg"])#établir la connection entre TraCI et SUMO
    run()

