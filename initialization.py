import settings
def initialization(Ni,vocab_size,no_of_events,levels):
  settings.N=Ni
  settings.V=vocab_size
  settings.K=no_of_events
  settings.H=levels
  settings.kappa
  settings.p=[]
  for pc in range(0,settings.K):
    settings.p.append(0)

  settings.pb=[]
  for pbc in range(0,settings.H):
    settings.pb.append(0)
  
  settings.pg=[]
  for pgc in range(settings.K):
    settings.pg.append(0)
    
  settings.p_phikw=[]
  for p_pc in range(0,settings.V):
    settings.p_phikw.append(0)
    
  settings.theta=[]
  settings.BETA=[]

  settings.tau=[]
  for ta in range(settings.H):
    settings.tau.append(0.01)

  settings.iota=[]
  for ia in range(settings.H):
    settings.iota.append(0.1)
    
  if settings.image_dataset_flag:
    settings.upsilon=1.0 #for phi
  else:
    settings.upsilon=1.0 #for phi

  print("N=",settings.N,"H=",settings.H,"K=",settings.K,"V=",settings.V,"mew=",settings.mew,"delta=",settings.delta,"kappa=",settings.kappa,"nu=",settings.nu,"lambda=",settings.lamb)
