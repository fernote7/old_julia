using JuMP
using MathProgBase
using Gurobi
using Distributions
using DataFrames


pld = readtable("C:\\Users\\leirb_000\\Dropbox\\Nova pasta\\Artigo\\pld1.csv",separator=';')
vento = readtable("C:\\Users\\leirb_000\\Dropbox\\Nova pasta\\Artigo\\vento.csv",separator=';')
srand(12345)
S=1000

vento1=Float64[vento[1,j] for j in 1:S]
pld1=Float64[pld[1,j] for j in 1:S]

mean(vento1)
std(vento1)
maximum(vento1)
minimum(vento1)

pot=wind_to_potency(vento1,tabela)
mean(pot)
std(pot*28)
maximum(pot*28)
minimum(pot*28)
###potencia gerada pelas 27 torres
G=28*pot*24*30
mean(G)
std(G)
maximum(G)
minimum(G)
###preco a termo utilizado
Pt=mean(pld1)*rand(Uniform(0.8,1.2),1)[1]

#preço do mwm no contrato
Pc=mean(pld1)*rand(Uniform(0.6,1.5),1)[1]

#Energia contratada
Qc=mean(G)
mean(Pc*Qc+pld1.*(G-Qc))
std(Pc*Qc+pld1.*(G-Qc))
maximum(Pc*Qc+pld1.*(G-Qc))
minimum(Pc*Qc+pld1.*(G-Qc))
sort(Pc*Qc+pld1.*(G-Qc))[50]
mean(sort(Pc*Qc+pld1.*(G-Qc))[1:50])

alpha=0.05
lambda=0.1
m=Model(solver=GurobiSolver())
@defVar(m,Qt)
@defVar(m,delta[1:S]>=0)
@defVar(m,R[1:S])
@defVar(m,z)
@setObjective(m,Max,(1-lambda)*(1/S)*sum{R[s],s=1:S}+(lambda)*(z-(1/(S*alpha))*sum{delta[s], s=1:S}))

@addConstraints(m,begin
                  sum_to_one[s=1:S], Pc*Qc-Pt*Qt+pld1[s]*(Qt+G[s]-Qc)==R[s]
                end)

@addConstraints(m,begin
                  sum_to_one[s=1:S], delta[s]>=z-R[s]
                end)

status=solve(m)

obj=getObjectiveValue(m)
result_Qt=getValue(Qt)
result_z=getValue(z)
result_R=getValue(R)

Result_R=Float64[result_R[i]for i in 1:S]
mean(Result_R)
sort(Result_R)[50]
mean(sort(Result_R)[1:50])
std(Result_R)
maximum(Result_R)
minimum(Result_R)
##########modelo para 12 meses
T=12
S=1000

vento1=Float64[vento[t,s] for t in 1:T, s in 1:S]
pld1=Float64[pld[t,s] for t in 1:T, s in 1:S]
mean(vento1)
std(vento1)
maximum(vento1)
minimum(vento1)


pot=Array(Float64,12,1000)
for i in 1:T
pot[i,:]=wind_to_potency(vento1[i,:],tabela)
end
mean(pot*28)
std(pot*28)
maximum(pot*28)
minimum(pot*28)

###potencia gerada pelas 28 torres
G=28*pot*24*30

###preco a termo utilizado
Pt=Array(Float64,T)
for i in 1:T
  Pt[i]=mean(pld1[i,:])*rand(Uniform(0.8,1.2),1)[1]
end
typeof(Pt)


#preço do mwm no contrato
Pc=Array(Float64,T)
for i in 1:T
  Pc[i]=mean(pld1[i,:])*rand(Uniform(0.5,1.5),1)[1]
end
typeof(Pc)


#Energia contratada
Qc=Array(Float64,T)
for i in 1:T
  Qc[i]=mean(G[i,:])
end
typeof(Qc)
lucrosc=Array(Float64,1000)
for(i in 1:1000)
  lucrosc[i]=sum(Pc.*Qc+pld1[:,i].*(G[:,i]-Qc))
end
mean(lucrosc)
std(lucrosc)
maximum(lucrosc)
minimum(lucrosc)
sort(lucrosc)[50]
mean(sort(lucrosc)[1:50])

lqc=(Pc.*Qc.+pld1.*(G.-Qc))
lucrom=Array(Float64,12)
for(i in 1:12)
  lucrom[i]=minimum(lqc[i,:])
end
println(lucrom)
lucroo=hcat(lucroo,lucrom)
writedlm("C:\\Users\\leirb_000\\Dropbox\\EYDI_MARCELO_GABRIEL\\modelo de otimização\\lucro1.csv",lucroo,";")

lb=0.5*Qc
k=0.05*mean(pld1)
lambda=0.5
alpha=0.05
m=Model(solver=GurobiSolver())
@defVar(m,Qt[1:T])
@defVar(m,delta[1:T,1:S])
@defVar(m,R[1:T,1:S])
@defVar(m,z)
@defVar(m,tcost[1:T])
@setObjective(m,Max,(1-lambda)*(1/S)*sum{R[t,s],t=1:T, s=1:S}+lambda*(z-(1/(S*alpha))*sum{delta[t,s], t=1:T, s=1:S}))

@addConstraints(m,begin
                  sum_to_one[t=1:T, s=1:S], Pc[t]*Qc[t]-Pt[t]*Qt[t]+pld1[t,s]*(Qt[t]+G[t,s]-Qc[t])-tcost[t]==R[t,s]
                end)

@addConstraints(m,begin
                  sum_to_one[s=1:S], sum{delta[t,s],t=1:T}>=z-sum{R[t,s],t=1:T}
                end)

@addConstraints(m,begin
                  sum_to_one[s=1:S], sum{delta[t,s], t=1:T}>=0
                end)

@addConstraints(m,begin
                  sum_to_one[t=1:T], Qt[t]>=-lb[t]
                end)

@addConstraints(m,begin
                  sum_to_one[t=1:T], tcost[t]>=k*Qt[t]
                end)

@addConstraints(m,begin
                  sum_to_one[t=1:T], tcost[t]>=-k*Qt[t]
                end)

status=solve(m)

obj=getObjectiveValue(m)
result_Qt=getValue(Qt)
result_z=getValue(z)
result_R=getValue(R)
println(result_Qt)
println(Qc)
Result_R=Float64[result_R[t,s]for t in 1:T, s in 1:S]
mean(Result_R)
aux=vec(Result_R)
sort(aux)[alpha*S*T]
mean(sort(aux)[1:alpha*S*T])

lucro1=Array(Float64,1000)
for(i in 1:1000)
  lucro1[i]=sum(Result_R[:,i])
end

mean(lucro1)
std(lucro1)
maximum(lucro1)
minimum(lucro1)
sort(lucro1)[50]
mean(sort(lucro1)[1:50])

lucrom=Array(Float64,12)
for(i in 1:12)
  lucrom[i]=mean(sort(Result_R[i,:][1:1000])[1:50])
end
println(lucrom)

lucroo=hcat(lucroo,lucrom)
writedlm("C:\\Users\\leirb_000\\Dropbox\\EYDI_MARCELO_GABRIEL\\modelo de otimização\\lucro1.csv",lucroo,";")


##########modelo para 12 meses com decisão de quantidade contratada
T=12
S=1000

vento1=Float64[vento[t,s] for t in 1:T, s in 1:S]
pld1=Float64[pld[t,s] for t in 1:T, s in 1:S]

pot=Array(Float64,12,1000)
for t in 1:T
pot[t,:]=wind_to_potency(vento1[t,:],tabela)
end
typeof(pot)
size(pot)

###potencia gerada pelas 27 torres
G=28*pot*24*30

###preco a termo utilizado
Pt=Array(Float64,T)
for i in 1:T
  Pt[i]=mean(pld1[i,:])*rand(Uniform(0.8,1.2),1)[1]
end
typeof(Pt)


#preço do mwm no contrato
Pc=Array(Float64,T)
for i in 1:T
  Pc[i]=mean(pld1[i,:])*rand(Uniform(0.5,1.5),1)[1]
end
typeof(Pc)

pld11=Array(Float64,12)
for i in 1:T
  pld11[i]=mean(pld1[i,:])[1]
end
println(pld11)




#Energia contratada
Qm=Array(Float64,T)
for i in 1:T
  Qm[i]=mean(G[i,:])
end
typeof(Qm)



k=0.05*mean(pld1)
lambda=0.99
alpha=0.05
m=Model(solver=GurobiSolver())
@defVar(m,Qt[1:T])
@defVar(m,delta[1:T,1:S])
@defVar(m,R[1:T,1:S])
@defVar(m,z)
@defVar(m,tcost[1:T])
@defVar(m,Qc[1:T]>=0)
@setObjective(m,Max,(1-lambda)*(1/S)*sum{R[t,s],t=1:T, s=1:S}+lambda*(z-(1/(S*alpha))*sum{delta[t,s], t=1:T, s=1:S}))

@addConstraints(m,begin
                  sum_to_one[t=1:T, s=1:S], Pc[t]*Qc[t]-Pt[t]*Qt[t]+pld1[t,s]*(Qt[t]+G[t,s]-Qc[t])-tcost[t]==R[t,s]
                end)

@addConstraints(m,begin
                  sum_to_one[s=1:S], sum{delta[t,s],t=1:T}>=z-sum{R[t,s],t=1:T}
                end)

@addConstraints(m,begin
                  sum_to_one[s=1:S], sum{delta[t,s], t=1:T}>=0
                end)

@addConstraints(m,begin
                  sum_to_one[t=1:T], Qt[t]>=-lb[t]
                end)

@addConstraints(m,begin
                  sum_to_one[t=1:T], tcost[t]>=k*Qt[t]
                end)

@addConstraints(m,begin
                  sum_to_one[t=1:T], tcost[t]>=-k*Qt[t]
                end)


@addConstraints(m,begin
                  sum_to_one[t=1:T], Qc[t]<=1.2*Qm[t]
                end)

@addConstraints(m,begin
                  sum_to_one[t=1:T], Qc[t]>=0.5*Qm[t]
                end)

status=solve(m)

obj=getObjectiveValue(m)
result_Qt=getValue(Qt)
result_z=getValue(z)
result_R=getValue(R)
result_Qc=getValue(Qc)

println(result_Qt)
println(result_Qc)

Result_R=Float64[result_R[t,s]for t in 1:T, s in 1:S]
mean(Result_R)
aux=vec(Result_R)
sort(aux)[alpha*S*T]
mean(sort(aux)[1:alpha*S*T])

lucro1=Array(Float64,1000)
for(i in 1:1000)
  lucro1[i]=sum(Result_R[:,i])
end

mean(lucro1)
std(lucro1)
maximum(lucro1)
minimum(lucro1)
sort(lucro1)[50]
mean(sort(lucro1)[1:50])

lucrom=Array(Float64,12)
for(i in 1:12)
  lucrom[i]=mean(sort(Result_R[i,:][1:1000])[1:50])
end
println(lucrom)

lucroo=hcat(lucroo,lucrom)
writedlm("C:\\Users\\leirb_000\\Dropbox\\EYDI_MARCELO_GABRIEL\\modelo de otimização\\lucro1.csv",lucroo,";")
