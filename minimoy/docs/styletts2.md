> Source: https://arxiv.org/abs/2306.05765

# On change of slow variables at crossing the separatrices 

Anatoly Neishtadt 

###### Abstract

We consider general (not necessarily Hamiltonian) perturbations of Hamiltonian systems with one degree of freedom near separatrices of the unperturbed system. We present asymptotic formulas for change of slow variables at evolution across separatrices.

##  1 Outline of the problem

We consider systems described by differential equations of the form

| q˙˙𝑞\displaystyle\dot{q} | =\displaystyle= | ∂H∂p+ε​fq,p˙=−∂H∂q+ε​fp,z˙=ε​fz,formulae-sequence𝐻𝑝𝜀subscript𝑓𝑞˙𝑝𝐻𝑞𝜀subscript𝑓𝑝˙𝑧𝜀subscript𝑓𝑧\displaystyle\frac{\partial H}{\partial p}+\varepsilon f_{q},\,\dot{p}=-\frac{\partial H}{\partial q}+\varepsilon f_{p},\,\dot{z}=\varepsilon f_{z}\,, |  | (1.1)  
---|---|---|---|---|---  
| H𝐻\displaystyle H | =\displaystyle= | H​(p,q,z),fα=fα​(p,q,z,ε),α=p,q,z,(p,q)∈ℝ2,z∈ℝl−2.formulae-sequence𝐻𝑝𝑞𝑧subscript𝑓𝛼subscript𝑓𝛼𝑝𝑞𝑧𝜀formulae-sequence𝛼𝑝𝑞𝑧formulae-sequence𝑝𝑞superscriptℝ2𝑧superscriptℝ𝑙2\displaystyle H(p,q,z),\,f_{\alpha}=f_{\alpha}(p,q,z,\varepsilon),\alpha=p,q,z,\,(p,q)\in\mathbb{R}^{2},z\in\mathbb{R}^{l-2}\,.\  |   
  
Here ε𝜀\varepsilon is a small parameter, |ε|≪1much-less-than𝜀1|\varepsilon|\ll 1. For ε=0,z=constformulae-sequence𝜀0𝑧const\varepsilon=0,\,z={\rm const} we have an unperturbed system for p,q𝑝𝑞p,q, which is a Hamiltonian system with one degree of freedom. The function H𝐻H is an unperturbed Hamiltonian. For ε>0𝜀0\varepsilon>0 we have a perturbed system, and functions ε​fα𝜀subscript𝑓𝛼\varepsilon f_{\alpha} are perturbations.

It is supposed that there are a saddle point and passing through it separatrices in the phase portrait of the unperturbed system, Fig. 1. Under the action of perturbations the projection of the phase point onto the plane (p,q)𝑝𝑞(p,q) crosses a separatrix.

Figure 1: Phase portrait of the unperturbed system.

Separatrices divide the phase plane of the unperturbed systems into domains G1​(z),G2​(z),G3​(z)subscript𝐺1𝑧subscript𝐺2𝑧subscript𝐺3𝑧G_{1}(z),G_{2}(z),G_{3}(z), Fig. 1. In each of these domains it is possible to use variables h,φℎ𝜑h,\varphi instead of p,q𝑝𝑞p,q, where hℎh is the difference between H𝐻H and its value at the saddle point, and φ𝜑\varphi is “the angle” (from the pair “action-angle” variables [1] of the unperturbed system). Then for h,z,φℎ𝑧𝜑h,z,\varphi we get the perturbed system having the standard form of system with one rotating phase [2]: in this system h,zℎ𝑧h,z are called slow variables, φ𝜑\varphi is the rotating phase. It is a classical result that the averaged with respect to φ𝜑\varphi system describes the evolution of h,zℎ𝑧h,z far from separatrices with accuracy O​(ε)𝑂𝜀O(\varepsilon) during the time interval of order 1/ε1𝜀1/\varepsilon [2]. For approximate description of evolution of h,zℎ𝑧h,z for trajectories that cross the unperturbed separatrices one can use the averaged system up to the separatrix and then averaged system with initial conditions on the separatrix in one of domains in which the trajectory is captured (a certain probability can be assigned to each such continuation). For majority of initial conditions this procedure describes the behaviour of slow variables with accuracy O​(ε​ln⁡ε)𝑂𝜀𝜀O(\varepsilon\ln\varepsilon) during time of order 1/ε1𝜀1/\varepsilon; the measure of the “bad” set of initial conditions, for which this description is not valid, tends to 00 faster than any given power of ε𝜀\varepsilon as ε→0→𝜀0\varepsilon\to 0 [10]. One can make one more step of the averaging method and use the same procedure for the second order averaged system (it is shown in [11] that solutions of this system indeed arrive to separatrices). This improves accuracy up to O​(ε2)𝑂superscript𝜀2O(\varepsilon^{2}) for motions far from separatrices. However, for motion with separatrix crossing there is no improvement. The reason is that there is a change of order at least ε𝜀\varepsilon of slow variables at crossing a narrow neighbourhood of separatrices. Because the width of this neighbourhood tends to 0 as ε→0→𝜀0\varepsilon\to 0, it is reasonable to call this change a jump of slow variables at the separatrix. In this note we give asymptotic formulas for this jump. Such formulas were first obtained in [12] for the pendulum in a slowly varying gravitational field, then in [5, 7] for the general case of a Hamiltonian system with one degree of freedom and slowly varying parameters, in [8] for the general case of a slow-fast Hamiltonian system with two degrees of freedom, and in [3, 4] for motion in a slowly time-dependent potential with a dissipation. Jump of slow variables is interpreted as a jump of an adiabatic invariant for Hamiltonian systems [12, 5, 7, 8] and as a time shift for systems with a dissipation [3, 4]. We consider the case of general perturbed system (1.1). For derivation of intermediate estimates used in this note see, e.g., [10, 11].

##  2 Asymptotic expansions for unperturbed motions near separatrices

In the phase portrait of the unperturbed system there is a saddle point C=C​(z)𝐶𝐶𝑧C=C(z) and passing through it separatrices l1=l1​(z),l2=l2​(z)formulae-sequencesubscript𝑙1subscript𝑙1𝑧subscript𝑙2subscript𝑙2𝑧l_{1}=l_{1}(z),l_{2}=l_{2}(z). We denote l3=l3​(z)=l1​(z)∪l2​(z)subscript𝑙3subscript𝑙3𝑧subscript𝑙1𝑧subscript𝑙2𝑧l_{3}=l_{3}(z)=l_{1}(z)\cup l_{2}(z). Denote qC=qC​(z),pC=pC​(z)formulae-sequencesubscript𝑞𝐶subscript𝑞𝐶𝑧subscript𝑝𝐶subscript𝑝𝐶𝑧q_{C}=q_{C}(z),p_{C}=p_{C}(z) coordinates of the point C𝐶C. Denote

| hC​(z)subscriptℎ𝐶𝑧\displaystyle h_{C}(z) | =H​(pC​(z),qC​(z),z),E​(p,q,z)=H​(p,q,z)−hC​(z).formulae-sequenceabsent𝐻subscript𝑝𝐶𝑧subscript𝑞𝐶𝑧𝑧𝐸𝑝𝑞𝑧𝐻𝑝𝑞𝑧subscriptℎ𝐶𝑧\displaystyle=H(p_{C}(z),q_{C}(z),z),\ E(p,q,z)=H(p,q,z)-h_{C}(z). |   
---|---|---|---  
  
We assume that E>0𝐸0E>0 in G3subscript𝐺3G_{3}, E<0𝐸0E<0 in G1,2subscript𝐺12G_{1,2}.

Denote

| fz,C(z\displaystyle f_{z,C}(z | )=fz(pC(z),qC(z),z,0),Fz(p,q,z)=f(p,q,z,0)−fz,C(z),\displaystyle)=f_{z}(p_{C}(z),q_{C}(z),z,0),\ F_{z}(p,q,z)=f(p,q,z,0)-f_{z,C}(z), |   
---|---|---|---  
| fh​(p,q,z)subscript𝑓ℎ𝑝𝑞𝑧\displaystyle f_{h}(p,q,z) | =∂E∂p​fp​(p,q,z,0)+∂E∂q​fq​(p,q,z,0)+∂E∂z​fz​(p,q,z,0).absent𝐸𝑝subscript𝑓𝑝𝑝𝑞𝑧0𝐸𝑞subscript𝑓𝑞𝑝𝑞𝑧0𝐸𝑧subscript𝑓𝑧𝑝𝑞𝑧0\displaystyle=\frac{\partial E}{\partial p}f_{p}(p,q,z,0)+\frac{\partial E}{\partial q}f_{q}(p,q,z,0)+\frac{\partial E}{\partial z}f_{z}(p,q,z,0). |   
  
For the period T𝑇T of the trajectory E=h𝐸ℎE=h in domain Gisubscript𝐺𝑖G_{i} we have

| T=−ai​ln⁡|h|+bi+O​(h​ln⁡|h|),a1=a2=a,a3=2​a,b3=b1+b2.formulae-sequenceformulae-sequence𝑇subscript𝑎𝑖ℎsubscript𝑏𝑖𝑂ℎℎsubscript𝑎1subscript𝑎2𝑎formulae-sequencesubscript𝑎32𝑎subscript𝑏3subscript𝑏1subscript𝑏2\displaystyle T=-a_{i}\ln|h|+b_{i}+O(h\ln|h|),\ a_{1}=a_{2}=a,a_{3}=2a,\ b_{3}=b_{1}+b_{2}. |   
---|---|---  
  
Denote

| ∮lifh​(p,q,z)​𝑑t=−Θi​(z),∮liFz​(p,q,z)​𝑑t=Ai​(z).formulae-sequencesubscriptcontour-integralsubscript𝑙𝑖subscript𝑓ℎ𝑝𝑞𝑧differential-d𝑡subscriptΘ𝑖𝑧subscriptcontour-integralsubscript𝑙𝑖subscript𝐹𝑧𝑝𝑞𝑧differential-d𝑡subscript𝐴𝑖𝑧\oint_{l_{i}}f_{h}(p,q,z)dt=-\Theta_{i}(z),\ \oint_{l_{i}}F_{z}(p,q,z)dt=A_{i}(z). |   
---|---|---  
  
Then for integrals along the unperturbed phase trajectory E=h𝐸ℎE=h in the domain Gisubscript𝐺𝑖G_{i} we have

| ∮E=hfh​(p,q,z)​𝑑tsubscriptcontour-integral𝐸ℎsubscript𝑓ℎ𝑝𝑞𝑧differential-d𝑡\displaystyle\oint_{E=h}f_{h}(p,q,z)dt | =−Θi​(z)+O​(h​ln⁡|h|),absentsubscriptΘ𝑖𝑧𝑂ℎℎ\displaystyle=-\Theta_{i}(z)+O(h\ln|h|), |   
---|---|---|---  
| ∮E=hFz​(p,q,z)​𝑑tsubscriptcontour-integral𝐸ℎsubscript𝐹𝑧𝑝𝑞𝑧differential-d𝑡\displaystyle\oint_{E=h}F_{z}(p,q,z)dt | =Ai​(z)+O​(h​ln⁡|h|).absentsubscript𝐴𝑖𝑧𝑂ℎℎ\displaystyle=A_{i}(z)+O(h\ln|h|). |   
  
We assume that Θ1​(z)>0,Θ2​(z)>0formulae-sequencesubscriptΘ1𝑧0subscriptΘ2𝑧0\Theta_{1}(z)>0,\Theta_{2}(z)>0 for all considered values of z𝑧z.

Introduce the coordinate system C​ξ​η𝐶𝜉𝜂C\xi\eta as shown in Fig. 1. For initial points on the positive side of the axis C​η𝐶𝜂C\eta and integrals on the unperturbed phase trajectory E=h𝐸ℎE=h (i.e. in G3subscript𝐺3G_{3}) we have

| 1T​∫0T(t−T2)​fh​𝑑t1𝑇superscriptsubscript0𝑇𝑡𝑇2subscript𝑓ℎdifferential-d𝑡\displaystyle\frac{1}{T}\int_{0}^{T}(t-\frac{T}{2})f_{h}dt | =−a​ln⁡h​(Θ2−Θ1)/2+(Θ1​b2−Θ2​b1)/2+d3−2​a​ln⁡h+b3+O​(h),absent𝑎ℎsubscriptΘ2subscriptΘ12subscriptΘ1subscript𝑏2subscriptΘ2subscript𝑏12subscript𝑑32𝑎ℎsubscript𝑏3𝑂ℎ\displaystyle=-\frac{a\ln h(\Theta_{2}-\Theta_{1})/2+(\Theta_{1}b_{2}-\Theta_{2}b_{1})/2+d_{3}}{-2a\ln h+b_{3}}+O(\sqrt{h}\,), |   
---|---|---|---  
| 1T​∫0T(t−T2)​Fz​𝑑t1𝑇superscriptsubscript0𝑇𝑡𝑇2subscript𝐹𝑧differential-d𝑡\displaystyle\frac{1}{T}\int_{0}^{T}(t-\frac{T}{2})F_{z}dt | =−a​ln⁡h​(A1−A2)/2−(A1​b2−A2​b1)/2+g3−2​a​ln⁡h+b3+O​(h).absent𝑎ℎsubscript𝐴1subscript𝐴22subscript𝐴1subscript𝑏2subscript𝐴2subscript𝑏12subscript𝑔32𝑎ℎsubscript𝑏3𝑂ℎ\displaystyle=-\frac{a\ln h(A_{1}-A_{2})/2-(A_{1}b_{2}-A_{2}b_{1})/2+g_{3}}{-2a\ln h+b_{3}}+O(\sqrt{h}\,). |   
  
For initial points on the axis C​ξ𝐶𝜉C\xi and integrals on the unperturbed phase trajectory E=h𝐸ℎE=h in the domain Gi,i=1,2formulae-sequencesubscript𝐺𝑖𝑖12G_{i},i=1,2 we have

| 1T​∫0T(t−T2)​fh​𝑑t1𝑇superscriptsubscript0𝑇𝑡𝑇2subscript𝑓ℎdifferential-d𝑡\displaystyle\frac{1}{T}\int_{0}^{T}(t-\frac{T}{2})f_{h}dt | =−di−a​ln⁡|h|+b1+O​(|h|),absentsubscript𝑑𝑖𝑎ℎsubscript𝑏1𝑂ℎ\displaystyle=-\frac{d_{i}}{-a\ln|h|+b_{1}}+O(\sqrt{|h|}\,), |   
---|---|---|---  
| 1T​∫0T(t−T2)​Fz​𝑑t1𝑇superscriptsubscript0𝑇𝑡𝑇2subscript𝐹𝑧differential-d𝑡\displaystyle\frac{1}{T}\int_{0}^{T}(t-\frac{T}{2})F_{z}dt | =−gi−a​ln⁡|h|+bi+O​(|h|).absentsubscript𝑔𝑖𝑎ℎsubscript𝑏𝑖𝑂ℎ\displaystyle=-\frac{g_{i}}{-a\ln|h|+b_{i}}+O(\sqrt{|h|}\,). |   
  
We have d3=d1+d2,g3=g1+g2formulae-sequencesubscript𝑑3subscript𝑑1subscript𝑑2subscript𝑔3subscript𝑔1subscript𝑔2d_{3}=d_{1}+d_{2},g_{3}=g_{1}+g_{2}.

In line with the general approach of the averaging method, one can make a change of variables

| h=h¯+ε​uh,1​(h¯,z¯,φ¯)+ε2​uh,2​(h¯,z¯,φ¯),z=z¯+ε​uz,1​(h¯,z¯,φ¯)+ε2​uz,2​(h¯,z¯,φ¯),φ=φ¯+ε​uφ,1​(h¯,z¯,φ¯)formulae-sequenceℎ¯ℎ𝜀subscript𝑢ℎ1¯ℎ¯𝑧¯𝜑superscript𝜀2subscript𝑢ℎ2¯ℎ¯𝑧¯𝜑formulae-sequence𝑧¯𝑧𝜀subscript𝑢𝑧1¯ℎ¯𝑧¯𝜑superscript𝜀2subscript𝑢𝑧2¯ℎ¯𝑧¯𝜑𝜑¯𝜑𝜀subscript𝑢𝜑1¯ℎ¯𝑧¯𝜑\displaystyle\begin{split}h&=\overline{h}+\varepsilon u_{h,1}(\overline{h},\overline{z},\overline{\varphi})+\varepsilon^{2}u_{h,2}(\overline{h},\overline{z},\overline{\varphi}),\\\ z&=\overline{z}+\varepsilon u_{z,1}(\overline{h},\overline{z},\overline{\varphi})+\varepsilon^{2}u_{z,2}(\overline{h},\overline{z},\overline{\varphi}),\\\ \varphi&=\overline{\varphi}+\varepsilon u_{\varphi,1}(\overline{h},\overline{z},\overline{\varphi})\end{split} |  | (2.1)  
---|---|---|---  
  
that transforms original equations of motion to the following form:

| h¯˙=ε​f¯h,1​(h¯,z¯)+ε2​f¯h,2​(h¯,z¯)+ε3​f¯h,3​(h¯,z¯,φ¯,ε),z¯˙=ε​f¯z,1​(h¯,z¯)+ε2​f¯z,2​(h¯,z¯)+ε3​f¯z,3​(h¯,z¯,φ¯,ε),φ¯˙=ω​(h¯,z¯)+ε​f¯φ,1​(h¯,z¯)+ε2​f¯φ,2​(h¯,z¯,φ¯,ε).formulae-sequence˙¯ℎ𝜀subscript¯𝑓ℎ1¯ℎ¯𝑧superscript𝜀2subscript¯𝑓ℎ2¯ℎ¯𝑧superscript𝜀3subscript¯𝑓ℎ3¯ℎ¯𝑧¯𝜑𝜀formulae-sequence˙¯𝑧𝜀subscript¯𝑓𝑧1¯ℎ¯𝑧superscript𝜀2subscript¯𝑓𝑧2¯ℎ¯𝑧superscript𝜀3subscript¯𝑓𝑧3¯ℎ¯𝑧¯𝜑𝜀˙¯𝜑𝜔¯ℎ¯𝑧𝜀subscript¯𝑓𝜑1¯ℎ¯𝑧superscript𝜀2subscript¯𝑓𝜑2¯ℎ¯𝑧¯𝜑𝜀\displaystyle\begin{split}\dot{\overline{h}}&=\varepsilon\overline{f}_{h,1}(\overline{h},\overline{z})+\varepsilon^{2}\overline{f}_{h,2}(\overline{h},\overline{z})+\varepsilon^{3}\overline{f}_{h,3}(\overline{h},\overline{z},\overline{\varphi},\varepsilon),\\\ \dot{\overline{z}}&=\varepsilon\overline{f}_{z,1}(\overline{h},\overline{z})+\varepsilon^{2}\overline{f}_{z,2}(\overline{h},\overline{z})+\varepsilon^{3}\overline{f}_{z,3}(\overline{h},\overline{z},\overline{\varphi},\varepsilon),\\\ \dot{\overline{\varphi}}&=\omega(\overline{h},\overline{z})+\varepsilon\overline{f}_{\varphi,1}(\overline{h},\overline{z})+\varepsilon^{2}\overline{f}_{\varphi,2}(\overline{h},\overline{z},\overline{\varphi},\varepsilon).\end{split} |  | (2.2)  
---|---|---|---  
  
The first order averaged system is obtained by keeping only the first term in each of these equations. The second order averaged system is obtained by neglecting highest order terms in each of these equations.

One can show that (see [11])

| uh,1=1T​∫0T(t−T2)​fh​𝑑t,uz,1=1T​∫0T(t−T2)​Fz​𝑑t.formulae-sequencesubscript𝑢ℎ11𝑇superscriptsubscript0𝑇𝑡𝑇2subscript𝑓ℎdifferential-d𝑡subscript𝑢𝑧11𝑇superscriptsubscript0𝑇𝑡𝑇2subscript𝐹𝑧differential-d𝑡u_{h,1}=\frac{1}{T}\int_{0}^{T}(t-\frac{T}{2})f_{h}dt,\ u_{z,1}=\frac{1}{T}\int_{0}^{T}(t-\frac{T}{2})F_{z}dt. |   
---|---|---  
  
It is convenient to consider evolution using both usual time t𝑡t and slow time ε​t𝜀𝑡\varepsilon t.

##  3 Jump of slow variables

###  3.1 General description of motion

Let a phase point start to move at t=t−=0𝑡subscript𝑡0t=t_{-}=0 (thus τ=τ−=0𝜏subscript𝜏0\tau=\tau_{-}=0) in the domain G3subscript𝐺3G_{3} at the distance of order 1 from the separatrix. Denote h−,z−,φ−subscriptℎsubscript𝑧subscript𝜑h_{-},z_{-},\varphi_{-} initial values of variables h,z,φℎ𝑧𝜑h,z,\varphi. Denote h​(t),z​(t),φ​(t)ℎ𝑡𝑧𝑡𝜑𝑡h(t),z(t),\varphi(t) solution of the system (1.1) with this initial condition (written in variables h,z,φℎ𝑧𝜑h,z,\varphi). The phase point makes rounds close to unperturbed trajectories in G3subscript𝐺3G_{3} while moving closer to the separatrix with each round, approaches the separatrix, crosses the separatrix and continues the motion in domain Gisubscript𝐺𝑖G_{i}, i=1𝑖1i=1 or i=2𝑖2i=2. Assume, for the sake of being definite, that this is motion in G2subscript𝐺2G_{2}. At t=t+=K/ε𝑡subscript𝑡𝐾𝜀t=t_{+}=K/\varepsilon (thus τ=τ+=K𝜏subscript𝜏𝐾\tau=\tau_{+}=K) the phase point is in G2subscript𝐺2G_{2} at the distance of order 1 form the separatrix. Here K=const𝐾constK={\rm const}. Denote h+=h​(t+),z+=z​(t+),φ+=φ​(t+)formulae-sequencesubscriptℎℎsubscript𝑡formulae-sequencesubscript𝑧𝑧subscript𝑡subscript𝜑𝜑subscript𝑡h_{+}=h(t_{+}),z_{+}=z(t_{+}),\varphi_{+}=\varphi(t_{+}).

Denote h¯​(τ),z¯​(τ)¯ℎ𝜏¯𝑧𝜏\overline{h}(\tau),\overline{z}(\tau) the solution of the first order averaged system with initial conditions h−,z−subscriptℎsubscript𝑧h_{-},z_{-} glued of solutions of averaged systems for domains G3subscript𝐺3G_{3} and G2subscript𝐺2G_{2} (cf. [10]). Denote τ∗subscript𝜏\tau_{*} the moment of the slow time such that h¯​(τ∗)=0¯ℎsubscript𝜏0\overline{h}(\tau_{*})=0 (i.e. τ∗subscript𝜏\tau_{*} is the moment of the slow time for the arrival of this solution to the separatrix). Denote z∗=z¯​(τ∗)subscript𝑧¯𝑧subscript𝜏z_{*}=\overline{z}(\tau_{*}).

Denote h^−​(τ),z^−​(τ)subscript^ℎ𝜏subscript^𝑧𝜏\hat{h}_{-}(\tau),\hat{z}_{-}(\tau) the solution of the second order averaged system with initial, at τ=0𝜏0\tau=0, conditions corresponding to h−,z−,φ−subscriptℎsubscript𝑧subscript𝜑h_{-},z_{-},\varphi_{-} (i.e., these initial conditions are obtained from h−,z−,φ−subscriptℎsubscript𝑧subscript𝜑h_{-},z_{-},\varphi_{-} by transformation (2.1)). Denote h^+​(τ),z^+​(τ)subscript^ℎ𝜏subscript^𝑧𝜏\hat{h}_{+}(\tau),\hat{z}_{+}(\tau) the solution of the second order averaged system with initial, at τ=τ+𝜏subscript𝜏\tau=\tau_{+}\,, conditions corresponding to h+,z+,φ+subscriptℎsubscript𝑧subscript𝜑h_{+},z_{+},\varphi_{+}. We consider this solution for τ≤τ+𝜏subscript𝜏\tau\leq\tau_{+}. Denote τ^∗,∓subscript^𝜏minus-or-plus\hat{\tau}_{*,\mp} moments of arrival of these two solutions to the separatrix, h^∓​(τ∗,∓)=0subscript^ℎminus-or-plussubscript𝜏minus-or-plus0\hat{h}_{\mp}(\tau_{*,\mp})=0. Denote z^∗,∓=z^∓​(τ^∗,∓)subscript^𝑧minus-or-plussubscript^𝑧minus-or-plussubscript^𝜏minus-or-plus\hat{z}_{*,\mp}=\hat{z}_{\mp}(\hat{\tau}_{*,\mp}). Denote

| Δ​τ^∗=τ^∗,+−τ^∗,−,Δ​z^∗=z^∗,+−z^∗,−.formulae-sequenceΔsubscript^𝜏subscript^𝜏subscript^𝜏Δsubscript^𝑧subscript^𝑧subscript^𝑧\Delta\hat{\tau}_{*}=\hat{\tau}_{*,+}-\hat{\tau}_{*,-},\ \Delta\hat{z}_{*}=\hat{z}_{*,+}-\hat{z}_{*,-}. |  | (3.1)  
---|---|---|---  
  
We will call these values jumps of slow variables at the separatrix. To estimate these jumps, we will consider description of dynamics by the second order averaged system at approaching the separatrix (in G3subscript𝐺3G_{3}) and at moving away from the separatrix (in G2subscript𝐺2G_{2}).

For crossing from domain G3subscript𝐺3G_{3} to domain Gisubscript𝐺𝑖G_{i}, i=1,2𝑖12i=1,2, we use also notations τ^∗,3=τ^∗,−,τ^∗,i=τ^∗,+,z^∗,3=z^∗,−,z^∗,i=z^∗,+formulae-sequencesubscript^𝜏3subscript^𝜏formulae-sequencesubscript^𝜏𝑖subscript^𝜏formulae-sequencesubscript^𝑧3subscript^𝑧subscript^𝑧𝑖subscript^𝑧\hat{\tau}_{*,3}=\hat{\tau}_{*,-},\hat{\tau}_{*,i}=\hat{\tau}_{*,+},\hat{z}_{*,3}=\hat{z}_{*,-},\hat{z}_{*,i}=\hat{z}_{*,+}.

Values fz,C,Θi,Ai,ai,bi,di,gisubscript𝑓𝑧𝐶subscriptΘ𝑖subscript𝐴𝑖subscript𝑎𝑖subscript𝑏𝑖subscript𝑑𝑖subscript𝑔𝑖f_{z,C},\Theta_{i},A_{i},a_{i},b_{i},d_{i},g_{i} are taken at z=z∗𝑧subscript𝑧z=z_{*} in all expansions below.

###  3.2 Approaching the separatrix

Consider motion of the phase point in G3subscript𝐺3G_{3}. Projection of the phase point onto p,q𝑝𝑞p,q plane makes rounds close to unperturbed trajectories while moving closer to the separatrix with each round. This projection crosses the ray C​η𝐶𝜂C\eta on each such round when it moves close enough to the separatrix. We enumerate N+1𝑁1N+1 moments of time for these intersections starting with the last one: t0>t1>…>tN>0subscript𝑡0subscript𝑡1…subscript𝑡𝑁0t_{0}>t_{1}>\ldots>t_{N}>0. The moment of time tNsubscript𝑡𝑁t_{N} is chosen in such a way that for 0≤t≤tN0𝑡subscript𝑡𝑁0\leq t\leq t_{N} dynamics of h,zℎ𝑧h,z is described with a required (high enough) accuracy by the second order averaged system, while for tN≤t≤t0subscript𝑡𝑁𝑡subscript𝑡0t_{N}\leq t\leq t_{0} expansions near the separatrix can be used for description of motion because the phase point is close enough to the separatrix.

Denote h~​(t),z~​(t),φ~​(t)~ℎ𝑡~𝑧𝑡~𝜑𝑡\tilde{h}(t),\tilde{z}(t),\tilde{\varphi}(t) the result of transformation of solution h​(t),z​(t),φ​(t)ℎ𝑡𝑧𝑡𝜑𝑡h(t),z(t),\varphi(t) via formulas (2.1). Denote

| hn=h​(tn),zn=z​(tn),h^n=h^​(tn),z^n=z^​(tn),h~N=h~​(tN),z~N=z~​(tN).formulae-sequencesubscriptℎ𝑛ℎsubscript𝑡𝑛formulae-sequencesubscript𝑧𝑛𝑧subscript𝑡𝑛formulae-sequencesubscript^ℎ𝑛^ℎsubscript𝑡𝑛formulae-sequencesubscript^𝑧𝑛^𝑧subscript𝑡𝑛formulae-sequencesubscript~ℎ𝑁~ℎsubscript𝑡𝑁subscript~𝑧𝑁~𝑧subscript𝑡𝑁\displaystyle h_{n}=h(t_{n}),z_{n}=z(t_{n}),\hat{h}_{n}=\hat{h}(t_{n}),\hat{z}_{n}=\hat{z}(t_{n}),\tilde{h}_{N}=\tilde{h}(t_{N}),\tilde{z}_{N}=\tilde{z}(t_{N}). |  | (3.2)  
---|---|---|---  
  
Denote Uh​(t)=uh,1+ε​uh,2,Uz​(t)=uz,1+ε​uz,2formulae-sequencesubscript𝑈ℎ𝑡subscript𝑢ℎ1𝜀subscript𝑢ℎ2subscript𝑈𝑧𝑡subscript𝑢𝑧1𝜀subscript𝑢𝑧2U_{h}(t)=u_{h,1}+\varepsilon u_{h,2},\ U_{z}(t)=u_{z,1}+\varepsilon u_{z,2} where functions uh,i,uz,isubscript𝑢ℎ𝑖subscript𝑢𝑧𝑖u_{h,i},u_{z,i} are those in (2.1), and they are calculated at the point h~​(t),z~​(t),φ~​(t)~ℎ𝑡~𝑧𝑡~𝜑𝑡\tilde{h}(t),\tilde{z}(t),\tilde{\varphi}(t). Denote Uh,n=Uh​(tn),Uz,n=Uz​(tn)formulae-sequencesubscript𝑈ℎ𝑛subscript𝑈ℎsubscript𝑡𝑛subscript𝑈𝑧𝑛subscript𝑈𝑧subscript𝑡𝑛U_{h,n}=U_{h}(t_{n}),U_{z,n}=U_{z}(t_{n}).

We will use the symbol ≃similar-to-or-equals\simeq in approximate equalities without indication of accuracy of the approximation. We have

| zN=z~N+ε​Uz,N≃z^n+ε​Uz,N.subscript𝑧𝑁subscript~𝑧𝑁𝜀subscript𝑈𝑧𝑁similar-to-or-equalssubscript^𝑧𝑛𝜀subscript𝑈𝑧𝑁z_{N}=\tilde{z}_{N}+\varepsilon U_{z,N}\simeq\hat{z}_{n}+\varepsilon U_{z,N}. |   
---|---|---  
  
Then we have an identity

| z0subscript𝑧0\displaystyle z_{0} | ≃z^3,∗+(z^|h=h0−z^3,∗)+((z^|h=hN−z^|h=h0)−(zN−z0))+(z^N−z^|h=hN)+ε​Uz,N.similar-to-or-equalsabsentsubscript^𝑧3evaluated-at^𝑧ℎsubscriptℎ0subscript^𝑧3evaluated-at^𝑧ℎsubscriptℎ𝑁evaluated-at^𝑧ℎsubscriptℎ0subscript𝑧𝑁subscript𝑧0subscript^𝑧𝑁evaluated-at^𝑧ℎsubscriptℎ𝑁𝜀subscript𝑈𝑧𝑁\displaystyle\simeq\hat{z}_{3,*}+(\hat{z}|_{h=h_{0}}-\hat{z}_{3,*})+\left((\hat{z}|_{h=h_{N}}-\hat{z}|_{h=h_{0}})-(z_{N}-z_{0})\right)+(\hat{z}_{N}-\hat{z}|_{h=h_{N}})+\varepsilon U_{z,N}. |  | (3.3)  
---|---|---|---|---  
  
Estimate terms in this expression separately.

a) For (z^|h=h0−z^3,∗)evaluated-at^𝑧ℎsubscriptℎ0subscript^𝑧3(\hat{z}|_{h=h_{0}}-\hat{z}_{3,*}).

This value is the change of z^^𝑧\hat{z} from the moment of time when h^=0^ℎ0\hat{h}=0 till the moment of time when h^=h0^ℎsubscriptℎ0\hat{h}=h_{0}. In the principal approximation

| z^˙=ε​(fz,C+1T​A3),h^˙=−ε​1T​Θ3.formulae-sequence˙^𝑧𝜀subscript𝑓𝑧𝐶1𝑇subscript𝐴3˙^ℎ𝜀1𝑇subscriptΘ3\displaystyle\dot{\hat{z}}=\varepsilon(f_{z,C}+\frac{1}{T}A_{3}),\ \dot{\hat{h}}=-\varepsilon\frac{1}{T}\Theta_{3}. |   
---|---|---  
  
Hence

| d​z^d​h^=−1Θ3​(T​fz,C+A3)𝑑^𝑧𝑑^ℎ1subscriptΘ3𝑇subscript𝑓𝑧𝐶subscript𝐴3\frac{d\hat{z}}{d\hat{h}}=-\frac{1}{\Theta_{3}}(Tf_{z,C}+A_{3}) |   
---|---|---  
  
and

| z^|h=h0evaluated-at^𝑧ℎsubscriptℎ0\displaystyle\hat{z}|_{h=h_{0}} | −z^3,∗=−1Θ3​∫0h0(T​fz,C+A3)​𝑑h=−fz,CΘ3​∫0h0T​𝑑h−A3Θ3​h0subscript^𝑧31subscriptΘ3superscriptsubscript0subscriptℎ0𝑇subscript𝑓𝑧𝐶subscript𝐴3differential-dℎsubscript𝑓𝑧𝐶subscriptΘ3superscriptsubscript0subscriptℎ0𝑇differential-dℎsubscript𝐴3subscriptΘ3subscriptℎ0\displaystyle-\hat{z}_{3,*}=-\frac{1}{\Theta_{3}}\int_{0}^{h_{0}}(Tf_{z,C}+A_{3})dh=-\frac{f_{z,C}}{\Theta_{3}}\int_{0}^{h_{0}}Tdh-\frac{A_{3}}{\Theta_{3}}h_{0} |   
---|---|---|---  
|  | =−fz,CΘ3​∫0h0(−2​a​ln⁡h+b3)​𝑑h−A3Θ3​h0=−fz,CΘ3​[−2​a​(h0​ln⁡h0−h0)+b3​h0]−A3Θ3​h0.absentsubscript𝑓𝑧𝐶subscriptΘ3superscriptsubscript0subscriptℎ02𝑎ℎsubscript𝑏3differential-dℎsubscript𝐴3subscriptΘ3subscriptℎ0subscript𝑓𝑧𝐶subscriptΘ3delimited-[]2𝑎subscriptℎ0subscriptℎ0subscriptℎ0subscript𝑏3subscriptℎ0subscript𝐴3subscriptΘ3subscriptℎ0\displaystyle=-\frac{f_{z,C}}{\Theta_{3}}\int_{0}^{h_{0}}(-2a\ln h+b_{3})dh-\frac{A_{3}}{\Theta_{3}}h_{0}=-\frac{f_{z,C}}{\Theta_{3}}\left[-2a(h_{0}\ln h_{0}-h_{0})+b_{3}h_{0}\right]-\frac{A_{3}}{\Theta_{3}}h_{0}. |   
  
b) For ((z^|h=hN−z^|h=h0)−(zN−z0))evaluated-at^𝑧ℎsubscriptℎ𝑁evaluated-at^𝑧ℎsubscriptℎ0subscript𝑧𝑁subscript𝑧0\left((\hat{z}|_{h=h_{N}}-\hat{z}|_{h=h_{0}})-(z_{N}-z_{0})\right).

To calculate this term one can consider motion round by round, calculate differences between changes of z^^𝑧\hat{z} and z𝑧z on each round, and sum up these differences. For changes of h,zℎ𝑧h,z one can use

|  | hn+1−hn≃ε​Θ3,similar-to-or-equalssubscriptℎ𝑛1subscriptℎ𝑛𝜀subscriptΘ3\displaystyle h_{n+1}-h_{n}\simeq\varepsilon\Theta_{3}, |   
---|---|---|---  
|  | zn+1−zn≃−ε​fz,C​(−a2​ln⁡hn−a​ln⁡(hn+ε​Θ1)−a2​ln⁡hn+1+b3)−ε​A3.similar-to-or-equalssubscript𝑧𝑛1subscript𝑧𝑛𝜀subscript𝑓𝑧𝐶𝑎2subscriptℎ𝑛𝑎subscriptℎ𝑛𝜀subscriptΘ1𝑎2subscriptℎ𝑛1subscript𝑏3𝜀subscript𝐴3\displaystyle z_{n+1}-z_{n}\simeq-\varepsilon f_{z,C}\left(-\frac{a}{2}\ln h_{n}-a\ln(h_{n}+\varepsilon\Theta_{1})-\frac{a}{2}\ln h_{n+1}+b_{3}\right)-\varepsilon A_{3}. |   
  
The change of z^^𝑧\hat{z} is calculated as

| z^|h=hn+1−z^|h=hn≃−fz,CΘ3​∫hnhn+1(−2​a​ln⁡h+b3)​𝑑h−ε​A3.similar-to-or-equalsevaluated-at^𝑧ℎsubscriptℎ𝑛1evaluated-at^𝑧ℎsubscriptℎ𝑛subscript𝑓𝑧𝐶subscriptΘ3superscriptsubscriptsubscriptℎ𝑛subscriptℎ𝑛12𝑎ℎsubscript𝑏3differential-dℎ𝜀subscript𝐴3\hat{z}|_{h=h_{n+1}}-\hat{z}|_{h=h_{n}}\simeq-\frac{f_{z,C}}{\Theta_{3}}\int_{h_{n}}^{h_{n+1}}(-2a\ln h+b_{3})dh-\varepsilon A_{3}. |   
---|---|---  
  
Thus

|  | (zn+1−zn)−(z^|h=hn+1−z^|h=hn)subscript𝑧𝑛1subscript𝑧𝑛evaluated-at^𝑧ℎsubscriptℎ𝑛1evaluated-at^𝑧ℎsubscriptℎ𝑛\displaystyle(z_{n+1}-z_{n})-(\hat{z}|_{h=h_{n+1}}-\hat{z}|_{h=h_{n}}) |   
---|---|---|---  
|  | ≃a​fz,CΘ3​[∫hnhn+1(−2​ln⁡h)​𝑑h−ε​Θ3​(−12​ln⁡hn−a​ln⁡(hn+ε​Θ1)−12​ln⁡hn+1)]similar-to-or-equalsabsent𝑎subscript𝑓𝑧𝐶subscriptΘ3delimited-[]superscriptsubscriptsubscriptℎ𝑛subscriptℎ𝑛12ℎdifferential-dℎ𝜀subscriptΘ312subscriptℎ𝑛𝑎subscriptℎ𝑛𝜀subscriptΘ112subscriptℎ𝑛1\displaystyle\simeq a\frac{f_{z,C}}{\Theta_{3}}\left[\int_{h_{n}}^{h_{n+1}}(-2\ln h)dh-\varepsilon\Theta_{3}\left(-\frac{1}{2}\ln h_{n}-a\ln(h_{n}+\varepsilon\Theta_{1})-\frac{1}{2}\ln h_{n+1}\right)\right] |   
  
The expression in the square brackets is related to calculation of the integral of −ln⁡hℎ-\ln h by the trapezoidal method like in [7]. Thus, we can directly use the expression for change of an adiabatic invariant from [7]. This gives

|  | (z^|h=hN−z^|h=h0)−(zN−z0)evaluated-at^𝑧ℎsubscriptℎ𝑁evaluated-at^𝑧ℎsubscriptℎ0subscript𝑧𝑁subscript𝑧0\displaystyle(\hat{z}|_{h=h_{N}}-\hat{z}|_{h=h_{0}})-(z_{N}-z_{0}) |   
---|---|---|---  
|  | ≃2​ε​a​fz,C​[−12​ln⁡2​πΓ​(ξ3)​Γ​(ξ3+θ13)+ξ3+(−ξ3+12​θ23)​ln⁡ξ3]similar-to-or-equalsabsent2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]122𝜋Γsubscript𝜉3Γsubscript𝜉3subscript𝜃13subscript𝜉3subscript𝜉312subscript𝜃23subscript𝜉3\displaystyle\simeq 2\varepsilon af_{z,C}\left[-\frac{1}{2}\ln\frac{2\pi}{\Gamma(\xi_{3})\Gamma(\xi_{3}+\theta_{13})}+\xi_{3}+\left(-\xi_{3}+\frac{1}{2}\theta_{23}\right)\ln\xi_{3}\right] |   
|  | +ε​12​a​fz,C​(θ23−θ13)​(ln⁡hN−ln⁡h0).𝜀12𝑎subscript𝑓𝑧𝐶subscript𝜃23subscript𝜃13subscriptℎ𝑁subscriptℎ0\displaystyle+\varepsilon\frac{1}{2}af_{z,C}(\theta_{23}-\theta_{13})(\ln h_{N}-\ln h_{0}). |   
  
Here Γ​(⋅)Γ⋅\Gamma(\cdot) is the gamma function, ξ3=h0/Θ3subscript𝜉3subscriptℎ0subscriptΘ3\xi_{3}=h_{0}/\Theta_{3}, θi​j=Θi/Θjsubscript𝜃𝑖𝑗subscriptΘ𝑖subscriptΘ𝑗\theta_{ij}=\Theta_{i}/\Theta_{j}.

c) For (z^N−z^|h=hN)subscript^𝑧𝑁evaluated-at^𝑧ℎsubscriptℎ𝑁(\hat{z}_{N}-\hat{z}|_{h=h_{N}}).

We have h​(tN)=hNℎsubscript𝑡𝑁subscriptℎ𝑁h(t_{N})=h_{N}. Denote t^Nsubscript^𝑡𝑁\hat{t}_{N} the moment of time such that h^​(t^N)=hN^ℎsubscript^𝑡𝑁subscriptℎ𝑁\hat{h}(\hat{t}_{N})=h_{N}. Find t^N−tNsubscript^𝑡𝑁subscript𝑡𝑁\hat{t}_{N}-t_{N}.

We have

| h^​(t^N)=hN=h​(tN)=h~​(tN)+ε​Uh,N≃h^​(tN)+ε​Uh,N.^ℎsubscript^𝑡𝑁subscriptℎ𝑁ℎsubscript𝑡𝑁~ℎsubscript𝑡𝑁𝜀subscript𝑈ℎ𝑁similar-to-or-equals^ℎsubscript𝑡𝑁𝜀subscript𝑈ℎ𝑁\hat{h}(\hat{t}_{N})=h_{N}=h(t_{N})=\tilde{h}(t_{N})+\varepsilon U_{h,N}\simeq\hat{h}(t_{N})+\varepsilon U_{h,N}. |   
---|---|---  
  
Thus

| t^N−tN≃−1Θ3​T​Uh,N.similar-to-or-equalssubscript^𝑡𝑁subscript𝑡𝑁1subscriptΘ3𝑇subscript𝑈ℎ𝑁\hat{t}_{N}-t_{N}\simeq-\frac{1}{\Theta_{3}}TU_{h,N}. |   
---|---|---  
  
Value of T𝑇T is calculated at h=hNℎsubscriptℎ𝑁h=h_{N}. Then

| z^Nsubscript^𝑧𝑁\displaystyle\hat{z}_{N} | −z^|h=hN=z^​(tN)−z^​(t^N)≃ε​(fz,C+1T​A3)​(tN−t^N)≃ε​(fz,C+1T​A3)​1Θ3​T​Uh,Nevaluated-at^𝑧ℎsubscriptℎ𝑁^𝑧subscript𝑡𝑁^𝑧subscript^𝑡𝑁similar-to-or-equals𝜀subscript𝑓𝑧𝐶1𝑇subscript𝐴3subscript𝑡𝑁subscript^𝑡𝑁similar-to-or-equals𝜀subscript𝑓𝑧𝐶1𝑇subscript𝐴31subscriptΘ3𝑇subscript𝑈ℎ𝑁\displaystyle-\hat{z}|_{h=h_{N}}=\hat{z}(t_{N})-\hat{z}(\hat{t}_{N})\simeq\varepsilon\left(f_{z,C}+\frac{1}{T}A_{3}\right)(t_{N}-\hat{t}_{N})\simeq\varepsilon\left(f_{z,C}+\frac{1}{T}A_{3}\right)\frac{1}{\Theta_{3}}TU_{h,N} |   
---|---|---|---  
|  | ≃−ε​fz,CΘ3​(a2​(Θ2−Θ1)​ln⁡hN+(Θ1​b2−Θ2​b1)/2+d3)+ε​14​A3​(θ23−θ13).similar-to-or-equalsabsent𝜀subscript𝑓𝑧𝐶subscriptΘ3𝑎2subscriptΘ2subscriptΘ1subscriptℎ𝑁subscriptΘ1subscript𝑏2subscriptΘ2subscript𝑏12subscript𝑑3𝜀14subscript𝐴3subscript𝜃23subscript𝜃13\displaystyle\simeq-\varepsilon\frac{f_{z,C}}{\Theta_{3}}\left(\frac{a}{2}(\Theta_{2}-\Theta_{1})\ln h_{N}+(\Theta_{1}b_{2}-\Theta_{2}b_{1})/2+d_{3}\right)+\varepsilon\frac{1}{4}A_{3}(\theta_{23}-\theta_{13}). |   
  
d) For ε​Uz,N𝜀subscript𝑈𝑧𝑁\varepsilon U_{z,N}.

We have ε​Uz,N≃ε​(A1−A2)/4similar-to-or-equals𝜀subscript𝑈𝑧𝑁𝜀subscript𝐴1subscript𝐴24\varepsilon U_{z,N}\simeq\varepsilon(A_{1}-A_{2})/4.

Combining results of a) - d) we get from identity (3.3)

| z0subscript𝑧0\displaystyle z_{0} | ≃z^3,∗−fz,CΘ3​[−2​a​(h0​ln⁡h0−h0)+b3​h0]−A3Θ3​h0similar-to-or-equalsabsentsubscript^𝑧3subscript𝑓𝑧𝐶subscriptΘ3delimited-[]2𝑎subscriptℎ0subscriptℎ0subscriptℎ0subscript𝑏3subscriptℎ0subscript𝐴3subscriptΘ3subscriptℎ0\displaystyle\simeq\hat{z}_{3,*}-\frac{f_{z,C}}{\Theta_{3}}\left[-2a(h_{0}\ln h_{0}-h_{0})+b_{3}h_{0}\right]-\frac{A_{3}}{\Theta_{3}}h_{0} |  | (3.4)  
---|---|---|---|---  
|  | +2​ε​a​fz,C​[−12​ln⁡2​πΓ​(ξ3)​Γ​(ξ3+θ13)+ξ3+(−ξ3+12​θ23)​ln⁡ξ3]2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]122𝜋Γsubscript𝜉3Γsubscript𝜉3subscript𝜃13subscript𝜉3subscript𝜉312subscript𝜃23subscript𝜉3\displaystyle+2\varepsilon af_{z,C}\left[-\frac{1}{2}\ln\frac{2\pi}{\Gamma(\xi_{3})\Gamma(\xi_{3}+\theta_{13})}+\xi_{3}+\left(-\xi_{3}+\frac{1}{2}\theta_{23}\right)\ln\xi_{3}\right] |   
|  | −ε​12​a​fz,C​(θ23−θ13)​ln⁡h0𝜀12𝑎subscript𝑓𝑧𝐶subscript𝜃23subscript𝜃13subscriptℎ0\displaystyle-\varepsilon\frac{1}{2}af_{z,C}(\theta_{23}-\theta_{13})\ln h_{0} |   
|  | −ε​12​fz,C​((θ13​b2−θ23​b1)+2​d3Θ3)+ε​14​A3​(θ23−θ13)𝜀12subscript𝑓𝑧𝐶subscript𝜃13subscript𝑏2subscript𝜃23subscript𝑏12subscript𝑑3subscriptΘ3𝜀14subscript𝐴3subscript𝜃23subscript𝜃13\displaystyle-\varepsilon\frac{1}{2}f_{z,C}\left((\theta_{13}b_{2}-\theta_{23}b_{1})+2\frac{d_{3}}{\Theta_{3}}\right)+\varepsilon\frac{1}{4}A_{3}(\theta_{23}-\theta_{13}) |   
|  | +ε​(A1−A2)/4𝜀subscript𝐴1subscript𝐴24\displaystyle+\varepsilon(A_{1}-A_{2})/4 |   
|  | =z^3,∗−fz,CΘ3​[−2​a​h0​ln⁡(ε​Θ3)+b3​h0]−A3Θ3​h0absentsubscript^𝑧3subscript𝑓𝑧𝐶subscriptΘ3delimited-[]2𝑎subscriptℎ0𝜀subscriptΘ3subscript𝑏3subscriptℎ0subscript𝐴3subscriptΘ3subscriptℎ0\displaystyle=\hat{z}_{3,*}-\frac{f_{z,C}}{\Theta_{3}}\left[-2ah_{0}\ln(\varepsilon\Theta_{3})+b_{3}h_{0}\right]-\frac{A_{3}}{\Theta_{3}}h_{0} |   
|  | +2​ε​a​fz,C​[−12​ln⁡2​πΓ​(ξ3)​Γ​(ξ3+θ13)+12​θ23​ln⁡ξ3]2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]122𝜋Γsubscript𝜉3Γsubscript𝜉3subscript𝜃1312subscript𝜃23subscript𝜉3\displaystyle+2\varepsilon af_{z,C}\left[-\frac{1}{2}\ln\frac{2\pi}{\Gamma(\xi_{3})\Gamma(\xi_{3}+\theta_{13})}+\frac{1}{2}\theta_{23}\ln\xi_{3}\right] |   
|  | −ε​12​a​fz,C​(θ23−θ13)​ln⁡h0𝜀12𝑎subscript𝑓𝑧𝐶subscript𝜃23subscript𝜃13subscriptℎ0\displaystyle-\varepsilon\frac{1}{2}af_{z,C}(\theta_{23}-\theta_{13})\ln h_{0} |   
|  | −ε​12​fz,C​((θ13​b2−θ23​b1)+2​d3Θ3)+ε​14​A3​(θ23−θ13)𝜀12subscript𝑓𝑧𝐶subscript𝜃13subscript𝑏2subscript𝜃23subscript𝑏12subscript𝑑3subscriptΘ3𝜀14subscript𝐴3subscript𝜃23subscript𝜃13\displaystyle-\varepsilon\frac{1}{2}f_{z,C}\left((\theta_{13}b_{2}-\theta_{23}b_{1})+2\frac{d_{3}}{\Theta_{3}}\right)+\varepsilon\frac{1}{4}A_{3}(\theta_{23}-\theta_{13}) |   
|  | +ε​(A1−A2)/4.𝜀subscript𝐴1subscript𝐴24\displaystyle+\varepsilon(A_{1}-A_{2})/4. |   
  
###  3.3 Passage through the separatrix

We assume that k​ε≤ξ3≤θ23−k​ε𝑘𝜀subscript𝜉3subscript𝜃23𝑘𝜀k\sqrt{\varepsilon}\leq\xi_{3}\leq\theta_{23}-k\sqrt{\varepsilon}, where k𝑘k is a large enough constant. Then for t>t0𝑡subscript𝑡0t>t_{0} the phase point makes a round close to the separatrix l2subscript𝑙2l_{2} and arrives to the ray C​ξ𝐶𝜉C\xi in G2subscript𝐺2G_{2} (Fig. 1) at some moment of time t0′=t0+O​(ln⁡ε)subscriptsuperscript𝑡′0subscript𝑡0𝑂𝜀t^{\prime}_{0}=t_{0}+O(\ln\varepsilon). Denote h0′=h​(t0′),z0′=z​(t0′)formulae-sequencesuperscriptsubscriptℎ0′ℎsubscriptsuperscript𝑡′0superscriptsubscript𝑧0′𝑧subscriptsuperscript𝑡′0h_{0}^{\prime}=h(t^{\prime}_{0}),z_{0}^{\prime}=z(t^{\prime}_{0}). We have

| h0′≃h0−ε​Θ2,z0′≃z0+ε​fz,C​[−a2​ln⁡h0−a2​ln⁡(−h0′)+b2]+ε​A2.formulae-sequencesimilar-to-or-equalssuperscriptsubscriptℎ0′subscriptℎ0𝜀subscriptΘ2similar-to-or-equalssuperscriptsubscript𝑧0′subscript𝑧0𝜀subscript𝑓𝑧𝐶delimited-[]𝑎2subscriptℎ0𝑎2superscriptsubscriptℎ0′subscript𝑏2𝜀subscript𝐴2h_{0}^{\prime}\simeq h_{0}-\varepsilon\Theta_{2},\ z_{0}^{\prime}\simeq z_{0}+\varepsilon f_{z,C}\left[-\frac{a}{2}\ln h_{0}-\frac{a}{2}\ln(-h_{0}^{\prime})+b_{2}\right]+\varepsilon A_{2}. |  | (3.5)  
---|---|---|---  
  
Denote ξ2=(−h0′)/(ε​Θ2)≃(ε​Θ2−h0)/(ε​Θ2)=(ε​Θ2−ε​Θ3​ξ3)/(ε​Θ2)=1−(Θ3/Θ2)​ξ3subscript𝜉2superscriptsubscriptℎ0′𝜀subscriptΘ2similar-to-or-equals𝜀subscriptΘ2subscriptℎ0𝜀subscriptΘ2𝜀subscriptΘ2𝜀subscriptΘ3subscript𝜉3𝜀subscriptΘ21subscriptΘ3subscriptΘ2subscript𝜉3\xi_{2}=(-h_{0}^{\prime})/(\varepsilon\Theta_{2})\simeq(\varepsilon\Theta_{2}-h_{0})/(\varepsilon\Theta_{2})=(\varepsilon\Theta_{2}-\varepsilon\Theta_{3}\xi_{3})/(\varepsilon\Theta_{2})=1-(\Theta_{3}/\Theta_{2})\xi_{3}. Thus ξ3≃θ23​(1−ξ2)similar-to-or-equalssubscript𝜉3subscript𝜃231subscript𝜉2\xi_{3}\simeq\theta_{23}(1-\xi_{2}).

###  3.4 Moving away from the separatrix

For t>t0′𝑡superscriptsubscript𝑡0′t>t_{0}^{\prime} the projection of the phase point onto p,q𝑝𝑞p,q plane makes rounds close to unperturbed trajectories while moving farther away from the separatrix with each round. This projection crosses the ray C​ξ𝐶𝜉C\xi in G2subscript𝐺2G_{2} on each such round while it moves close enough to the separatrix. We enumerate N+1𝑁1N+1 moments of time for these intersections starting with the first one: t0′<t1′<…<tN′<K/εsuperscriptsubscript𝑡0′superscriptsubscript𝑡1′…superscriptsubscript𝑡𝑁′𝐾𝜀t_{0}^{\prime}<t_{1}^{\prime}<\ldots<t_{N}^{\prime}<K/\varepsilon. The moment of time tN′superscriptsubscript𝑡𝑁′t_{N}^{\prime} is chosen in such a way that for tN′≤t≤K/εsuperscriptsubscript𝑡𝑁′𝑡𝐾𝜀t_{N}^{\prime}\leq t\leq K/\varepsilon changes of h,zℎ𝑧h,z are described with a required (high enough) accuracy by the second order averaged system, while for t0′≤t≤tN′superscriptsubscript𝑡0′𝑡superscriptsubscript𝑡𝑁′t_{0}^{\prime}\leq t\leq t_{N}^{\prime} expansions near the separatrix can be used for description of motion because the phase point is close enough to the separatrix. Calculations here are similar to those for approaching the separatrix in Section 3.2. In what follows, we omit ‘prime’ in notation for moments of time and use for variables h,zℎ𝑧h,z the same notation as in Section 3.2, except of h0′,z0′superscriptsubscriptℎ0′superscriptsubscript𝑧0′h_{0}^{\prime},z_{0}^{\prime}.

We have

| zN=z~N+ε​Uz,N≃z^n+ε​Uz,N.subscript𝑧𝑁subscript~𝑧𝑁𝜀subscript𝑈𝑧𝑁similar-to-or-equalssubscript^𝑧𝑛𝜀subscript𝑈𝑧𝑁z_{N}=\tilde{z}_{N}+\varepsilon U_{z,N}\simeq\hat{z}_{n}+\varepsilon U_{z,N}. |   
---|---|---  
  
Then we have an identity

| z0′superscriptsubscript𝑧0′\displaystyle z_{0}^{\prime} | ≃z^2,∗+(z^|h=h0′−z^2,∗)+((z^|h=hN−z^|h=h0′)−(zN−z0′))+(z^N−z^|h=hN)+ε​Uz,N.similar-to-or-equalsabsentsubscript^𝑧2evaluated-at^𝑧ℎsuperscriptsubscriptℎ0′subscript^𝑧2evaluated-at^𝑧ℎsubscriptℎ𝑁evaluated-at^𝑧ℎsuperscriptsubscriptℎ0′subscript𝑧𝑁superscriptsubscript𝑧0′subscript^𝑧𝑁evaluated-at^𝑧ℎsubscriptℎ𝑁𝜀subscript𝑈𝑧𝑁\displaystyle\simeq\hat{z}_{2,*}+(\hat{z}|_{h=h_{0}^{\prime}}-\hat{z}_{2,*})+\left((\hat{z}|_{h=h_{N}}-\hat{z}|_{h=h_{0}^{\prime}})-(z_{N}-z_{0}^{\prime})\right)+(\hat{z}_{N}-\hat{z}|_{h=h_{N}})+\varepsilon U_{z,N}. |  | (3.6)  
---|---|---|---|---  
  
Estimate terms in this expression separately.

a) For (z^|h=h0−z^2,∗)evaluated-at^𝑧ℎsubscriptℎ0subscript^𝑧2(\hat{z}|_{h=h_{0}}-\hat{z}_{2,*}).

Similarly to Section 3.2 we get

| z^|h=h0′−z^2,∗≃−fz,CΘ2​[−a​(h0′​ln⁡|h0′|−h0′)+b2​h0′]−A2Θ2​h0′.similar-to-or-equalsevaluated-at^𝑧ℎsuperscriptsubscriptℎ0′subscript^𝑧2subscript𝑓𝑧𝐶subscriptΘ2delimited-[]𝑎superscriptsubscriptℎ0′superscriptsubscriptℎ0′superscriptsubscriptℎ0′subscript𝑏2superscriptsubscriptℎ0′subscript𝐴2subscriptΘ2superscriptsubscriptℎ0′\hat{z}|_{h=h_{0}^{\prime}}-\hat{z}_{2,*}\simeq-\frac{f_{z,C}}{\Theta_{2}}\left[-a(h_{0}^{\prime}\ln|h_{0}^{\prime}|-h_{0}^{\prime})+b_{2}h_{0}^{\prime}\right]-\frac{A_{2}}{\Theta_{2}}h_{0}^{\prime}. |  | (3.7)  
---|---|---|---  
  
b) For ((z^|h=hN−z^|h=h0′)−(zN−z0′))evaluated-at^𝑧ℎsubscriptℎ𝑁evaluated-at^𝑧ℎsuperscriptsubscriptℎ0′subscript𝑧𝑁superscriptsubscript𝑧0′\left((\hat{z}|_{h=h_{N}}-\hat{z}|_{h=h_{0}^{\prime}})-(z_{N}-z_{0}^{\prime})\right).

Similarly to Section 3.2 we can use result of [7]. This gives

| (z^|h=hN−z^|h=h0′)−(zN−z0′)≃−ε​a​fz,C​[−ln⁡2​πΓ​(ξ2)+ξ2+(12−ξ2)​ln⁡ξ2].similar-to-or-equalsevaluated-at^𝑧ℎsubscriptℎ𝑁evaluated-at^𝑧ℎsuperscriptsubscriptℎ0′subscript𝑧𝑁superscriptsubscript𝑧0′𝜀𝑎subscript𝑓𝑧𝐶delimited-[]2𝜋Γsubscript𝜉2subscript𝜉212subscript𝜉2subscript𝜉2(\hat{z}|_{h=h_{N}}-\hat{z}|_{h=h_{0}^{\prime}})-(z_{N}-z_{0}^{\prime})\simeq-\varepsilon af_{z,C}\left[-\ln\frac{\sqrt{2\pi}}{\Gamma(\xi_{2})}+\xi_{2}+(\frac{1}{2}-\xi_{2})\ln\xi_{2}\right]. |  | (3.8)  
---|---|---|---  
  
c) For (z^N−z^|h=hN)subscript^𝑧𝑁evaluated-at^𝑧ℎsubscriptℎ𝑁(\hat{z}_{N}-\hat{z}|_{h=h_{N}})

Similarly to Section 3.2 we get

| z^N−z^|h=hN≃−ε​fz,CΘ2​d2.similar-to-or-equalssubscript^𝑧𝑁evaluated-at^𝑧ℎsubscriptℎ𝑁𝜀subscript𝑓𝑧𝐶subscriptΘ2subscript𝑑2\hat{z}_{N}-\hat{z}|_{h=h_{N}}\simeq-\varepsilon\frac{f_{z,C}}{\Theta_{2}}d_{2}. |  | (3.9)  
---|---|---|---  
  
d) For ε​Uz,N𝜀subscript𝑈𝑧𝑁\varepsilon U_{z,N}.

We get ε​Uz,N≃0similar-to-or-equals𝜀subscript𝑈𝑧𝑁0\varepsilon U_{z,N}\simeq 0.

Combining results of a) - d) we get from identity (3.6)

| z0′superscriptsubscript𝑧0′\displaystyle z_{0}^{\prime} | ≃z^2,∗−fz,CΘ2​[−a​(h0′​ln⁡|h0′|−h0′)+b2​h0′]−A2Θ2​h0′similar-to-or-equalsabsentsubscript^𝑧2subscript𝑓𝑧𝐶subscriptΘ2delimited-[]𝑎superscriptsubscriptℎ0′superscriptsubscriptℎ0′superscriptsubscriptℎ0′subscript𝑏2superscriptsubscriptℎ0′subscript𝐴2subscriptΘ2superscriptsubscriptℎ0′\displaystyle\simeq\hat{z}_{2,*}-\frac{f_{z,C}}{\Theta_{2}}\left[-a(h_{0}^{\prime}\ln|h_{0}^{\prime}|-h_{0}^{\prime})+b_{2}h_{0}^{\prime}\right]-\frac{A_{2}}{\Theta_{2}}h_{0}^{\prime} |  | (3.10)  
---|---|---|---|---  
|  | −ε​a​fz,C​[−ln⁡2​πΓ​(ξ2)+ξ2+(12−ξ2)​ln⁡ξ2]−ε​fz,CΘ2​d2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]2𝜋Γsubscript𝜉2subscript𝜉212subscript𝜉2subscript𝜉2𝜀subscript𝑓𝑧𝐶subscriptΘ2subscript𝑑2\displaystyle-\varepsilon af_{z,C}\left[-\ln\frac{\sqrt{2\pi}}{\Gamma(\xi_{2})}+\xi_{2}+(\frac{1}{2}-\xi_{2})\ln\xi_{2}\right]-\varepsilon\frac{f_{z,C}}{\Theta_{2}}d_{2} |   
|  | =z^2,∗−ε​fz,C​[a​(ξ2​ln⁡(ε​Θ2​ξ2)−ξ2)−b2​ξ2]+ε​A2​ξ2absentsubscript^𝑧2𝜀subscript𝑓𝑧𝐶delimited-[]𝑎subscript𝜉2𝜀subscriptΘ2subscript𝜉2subscript𝜉2subscript𝑏2subscript𝜉2𝜀subscript𝐴2subscript𝜉2\displaystyle=\hat{z}_{2,*}-\varepsilon{f_{z,C}}\left[a(\xi_{2}\ln(\varepsilon\Theta_{2}\xi_{2})-\xi_{2})-b_{2}\xi_{2}\right]+\varepsilon{A_{2}}\xi_{2} |   
|  | −ε​a​fz,C​[−ln⁡2​πΓ​(ξ2)+ξ2+(12−ξ2)​ln⁡ξ2]−ε​fz,CΘ2​d2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]2𝜋Γsubscript𝜉2subscript𝜉212subscript𝜉2subscript𝜉2𝜀subscript𝑓𝑧𝐶subscriptΘ2subscript𝑑2\displaystyle-\varepsilon af_{z,C}\left[-\ln\frac{\sqrt{2\pi}}{\Gamma(\xi_{2})}+\xi_{2}+(\frac{1}{2}-\xi_{2})\ln\xi_{2}\right]-\varepsilon\frac{f_{z,C}}{\Theta_{2}}d_{2} |   
|  | =z^2,∗−ε​fz,C​[a​ξ2​ln⁡(ε​Θ2)−b2​ξ2]+ε​A2​ξ2absentsubscript^𝑧2𝜀subscript𝑓𝑧𝐶delimited-[]𝑎subscript𝜉2𝜀subscriptΘ2subscript𝑏2subscript𝜉2𝜀subscript𝐴2subscript𝜉2\displaystyle=\hat{z}_{2,*}-\varepsilon{f_{z,C}}\left[a\xi_{2}\ln(\varepsilon\Theta_{2})-b_{2}\xi_{2}\right]+\varepsilon{A_{2}}\xi_{2} |   
|  | −ε​a​fz,C​[−ln⁡2​πΓ​(ξ2)+12​ln⁡ξ2]−ε​fz,CΘ2​d2.𝜀𝑎subscript𝑓𝑧𝐶delimited-[]2𝜋Γsubscript𝜉212subscript𝜉2𝜀subscript𝑓𝑧𝐶subscriptΘ2subscript𝑑2\displaystyle-\varepsilon af_{z,C}\left[-\ln\frac{\sqrt{2\pi}}{\Gamma(\xi_{2})}+\frac{1}{2}\ln\xi_{2}\right]-\varepsilon\frac{f_{z,C}}{\Theta_{2}}d_{2}. |   
  
###  3.5 Formula for jump of slow variables

Combining results of Sections 3.2, 3.3, 3.4 (formulas (3.4), (3.5) and (3.10) ) we get

|  | z^2,∗−ε​fz,C​[a​ξ2​ln⁡(ε​Θ2)−b2​ξ2]+ε​A2​ξ2subscript^𝑧2𝜀subscript𝑓𝑧𝐶delimited-[]𝑎subscript𝜉2𝜀subscriptΘ2subscript𝑏2subscript𝜉2𝜀subscript𝐴2subscript𝜉2\displaystyle\hat{z}_{2,*}-\varepsilon{f_{z,C}}\left[a\xi_{2}\ln(\varepsilon\Theta_{2})-b_{2}\xi_{2}\right]+\varepsilon{A_{2}}\xi_{2} |  | (3.11)  
---|---|---|---|---  
|  | −ε​a​fz,C​[−ln⁡2​πΓ​(ξ2)+12​ln⁡ξ2]−ε​fz,CΘ2​d2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]2𝜋Γsubscript𝜉212subscript𝜉2𝜀subscript𝑓𝑧𝐶subscriptΘ2subscript𝑑2\displaystyle-\varepsilon af_{z,C}\left[-\ln\frac{\sqrt{2\pi}}{\Gamma(\xi_{2})}+\frac{1}{2}\ln\xi_{2}\right]-\varepsilon\frac{f_{z,C}}{\Theta_{2}}d_{2} |   
|  | ≃z^3,∗−fz,CΘ3​[−2​a​h0​ln⁡(ε​Θ3)+b3​h0]−A3Θ3​h0similar-to-or-equalsabsentsubscript^𝑧3subscript𝑓𝑧𝐶subscriptΘ3delimited-[]2𝑎subscriptℎ0𝜀subscriptΘ3subscript𝑏3subscriptℎ0subscript𝐴3subscriptΘ3subscriptℎ0\displaystyle\simeq\hat{z}_{3,*}-\frac{f_{z,C}}{\Theta_{3}}\left[-2ah_{0}\ln(\varepsilon\Theta_{3})+b_{3}h_{0}\right]-\frac{A_{3}}{\Theta_{3}}h_{0} |   
|  | +2​ε​a​fz,C​[−12​ln⁡2​πΓ​(ξ3)​Γ​(ξ3+θ13)+12​θ23​ln⁡ξ3]2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]122𝜋Γsubscript𝜉3Γsubscript𝜉3subscript𝜃1312subscript𝜃23subscript𝜉3\displaystyle+2\varepsilon af_{z,C}\left[-\frac{1}{2}\ln\frac{2\pi}{\Gamma(\xi_{3})\Gamma(\xi_{3}+\theta_{13})}+\frac{1}{2}\theta_{23}\ln\xi_{3}\right] |   
|  | −ε​12​a​fz,C​(θ23−θ13)​ln⁡h0𝜀12𝑎subscript𝑓𝑧𝐶subscript𝜃23subscript𝜃13subscriptℎ0\displaystyle-\varepsilon\frac{1}{2}af_{z,C}(\theta_{23}-\theta_{13})\ln h_{0} |   
|  | −ε​12​fz,C​((θ13​b2−θ23​b1)+2​d3Θ3)+ε​14​A3​(θ23−θ13)𝜀12subscript𝑓𝑧𝐶subscript𝜃13subscript𝑏2subscript𝜃23subscript𝑏12subscript𝑑3subscriptΘ3𝜀14subscript𝐴3subscript𝜃23subscript𝜃13\displaystyle-\varepsilon\frac{1}{2}f_{z,C}\left((\theta_{13}b_{2}-\theta_{23}b_{1})+2\frac{d_{3}}{\Theta_{3}}\right)+\varepsilon\frac{1}{4}A_{3}(\theta_{23}-\theta_{13}) |   
|  | +ε​(A1−A2)/4𝜀subscript𝐴1subscript𝐴24\displaystyle+\varepsilon(A_{1}-A_{2})/4 |   
|  | +ε​fz,C​[−a2​ln⁡h0−a2​ln⁡(−h0′)+b2]+ε​A2𝜀subscript𝑓𝑧𝐶delimited-[]𝑎2subscriptℎ0𝑎2superscriptsubscriptℎ0′subscript𝑏2𝜀subscript𝐴2\displaystyle+\varepsilon f_{z,C}\left[-\frac{a}{2}\ln h_{0}-\frac{a}{2}\ln(-h_{0}^{\prime})+b_{2}\right]+\varepsilon A_{2} |   
  
Therefore

|  | Δ​z^∗=z^∗,+−z^∗,−=z^2,∗−z^3,∗Δsubscript^𝑧subscript^𝑧subscript^𝑧subscript^𝑧2subscript^𝑧3\displaystyle\Delta\hat{z}_{*}=\hat{z}_{*,+}-\hat{z}_{*,-}=\hat{z}_{2,*}-\hat{z}_{3,*} |  | (3.12)  
---|---|---|---|---  
|  | ≃ε​fz,C​[a​ξ2​ln⁡(ε​Θ2)−b2​ξ2]−ε​A2​ξ2similar-to-or-equalsabsent𝜀subscript𝑓𝑧𝐶delimited-[]𝑎subscript𝜉2𝜀subscriptΘ2subscript𝑏2subscript𝜉2𝜀subscript𝐴2subscript𝜉2\displaystyle\simeq\varepsilon{f_{z,C}}\left[a\xi_{2}\ln(\varepsilon\Theta_{2})-b_{2}\xi_{2}\right]-\varepsilon{A_{2}}\xi_{2} |   
|  | ε​a​fz,C​[−ln⁡2​πΓ​(ξ2)+12​ln⁡ξ2]+ε​fz,CΘ2​d2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]2𝜋Γsubscript𝜉212subscript𝜉2𝜀subscript𝑓𝑧𝐶subscriptΘ2subscript𝑑2\displaystyle\varepsilon af_{z,C}\left[-\ln\frac{\sqrt{2\pi}}{\Gamma(\xi_{2})}+\frac{1}{2}\ln\xi_{2}\right]+\varepsilon\frac{f_{z,C}}{\Theta_{2}}d_{2} |   
|  | −fz,CΘ3​[−2​a​h0​ln⁡(ε​Θ3)+b3​h0]−A3Θ3​h0subscript𝑓𝑧𝐶subscriptΘ3delimited-[]2𝑎subscriptℎ0𝜀subscriptΘ3subscript𝑏3subscriptℎ0subscript𝐴3subscriptΘ3subscriptℎ0\displaystyle-\frac{f_{z,C}}{\Theta_{3}}\left[-2ah_{0}\ln(\varepsilon\Theta_{3})+b_{3}h_{0}\right]-\frac{A_{3}}{\Theta_{3}}h_{0} |   
|  | +2​ε​a​fz,C​[−12​ln⁡2​πΓ​(ξ3)​Γ​(ξ3+θ13)+12​θ23​ln⁡ξ3]2𝜀𝑎subscript𝑓𝑧𝐶delimited-[]122𝜋Γsubscript𝜉3Γsubscript𝜉3subscript𝜃1312subscript𝜃23subscript𝜉3\displaystyle+2\varepsilon af_{z,C}\left[-\frac{1}{2}\ln\frac{2\pi}{\Gamma(\xi_{3})\Gamma(\xi_{3}+\theta_{13})}+\frac{1}{2}\theta_{23}\ln\xi_{3}\right] |   
|  | −ε​12​a​fz,C​(θ23−θ13)​ln⁡h0𝜀12𝑎subscript𝑓𝑧𝐶subscript𝜃23subscript𝜃13subscriptℎ0\displaystyle-\varepsilon\frac{1}{2}af_{z,C}(\theta_{23}-\theta_{13})\ln h_{0} |   
|  | −ε​12​fz,C​((θ13​b2−θ23​b1)+2​d3Θ3)+ε​14​A3​(θ23−θ13)𝜀12subscript𝑓𝑧𝐶subscript𝜃13subscript𝑏2subscript𝜃23subscript𝑏12subscript𝑑3subscriptΘ3𝜀14subscript𝐴3subscript𝜃23subscript𝜃13\displaystyle-\varepsilon\frac{1}{2}f_{z,C}\left((\theta_{13}b_{2}-\theta_{23}b_{1})+2\frac{d_{3}}{\Theta_{3}}\right)+\varepsilon\frac{1}{4}A_{3}(\theta_{23}-\theta_{13}) |   
|  | +ε​(A1−A2)/4𝜀subscript𝐴1subscript𝐴24\displaystyle+\varepsilon(A_{1}-A_{2})/4 |   
|  | +ε​fz,C​[−a2​ln⁡h0−a2​ln⁡(−h0′)+b2]+ε​A2.𝜀subscript𝑓𝑧𝐶delimited-[]𝑎2subscriptℎ0𝑎2superscriptsubscriptℎ0′subscript𝑏2𝜀subscript𝐴2\displaystyle+\varepsilon f_{z,C}\left[-\frac{a}{2}\ln h_{0}-\frac{a}{2}\ln(-h_{0}^{\prime})+b_{2}\right]+\varepsilon A_{2}. |   
  
For passage form G3subscript𝐺3G_{3} to G1subscript𝐺1G_{1} we would have relation (LABEL:for_jump_0) with replacement of index ‘2’ by index ‘1’ . The final result in the form for passage from G3subscript𝐺3G_{3} to Gisubscript𝐺𝑖G_{i} where i=1​or​ 2𝑖1or2i=1\ {\rm or}\ 2 is simplified to

|  | Δ​z^∗=z^i,∗−z^3,∗≃ε​fz,C​a​(ξi−12)​(ln⁡(ε​Θi)−2​θi​3​ln⁡(ε​Θ3))Δsubscript^𝑧subscript^𝑧𝑖subscript^𝑧3similar-to-or-equals𝜀subscript𝑓𝑧𝐶𝑎subscript𝜉𝑖12𝜀subscriptΘ𝑖2subscript𝜃𝑖3𝜀subscriptΘ3\displaystyle\Delta\hat{z}_{*}=\hat{z}_{i,*}-\hat{z}_{3,*}\simeq\varepsilon{f_{z,C}}a(\xi_{i}-\frac{1}{2})(\ln(\varepsilon\Theta_{i})-2\theta_{i3}\ln(\varepsilon\Theta_{3})) |  | (3.13)  
---|---|---|---|---  
|  | −ε​a​fz,C​ln⁡(2​π)3/2Γ(ξi)Γ(θi​3(1−ξi)Γ(1−θi​3ξi)\displaystyle-\varepsilon af_{z,C}\ln{\frac{(2\pi)^{3/2}}{\Gamma(\xi_{i})\Gamma(\theta_{i3}(1-\xi_{i})\Gamma(1-\theta_{i3}\xi_{i})}} |   
|  | −ε​fz,C​(ξi−12)​(bi−θi​3​b3)−ε​(ξi−12)​(Ai−θi​3​A3)𝜀subscript𝑓𝑧𝐶subscript𝜉𝑖12subscript𝑏𝑖subscript𝜃𝑖3subscript𝑏3𝜀subscript𝜉𝑖12subscript𝐴𝑖subscript𝜃𝑖3subscript𝐴3\displaystyle-\varepsilon f_{z,C}(\xi_{i}-\frac{1}{2})(b_{i}-\theta_{i3}b_{3})-\varepsilon(\xi_{i}-\frac{1}{2})(A_{i}-\theta_{i3}A_{3}) |   
|  | +ε​fz,CΘi​(di−θi​3​d3).𝜀subscript𝑓𝑧𝐶subscriptΘ𝑖subscript𝑑𝑖subscript𝜃𝑖3subscript𝑑3\displaystyle+\varepsilon\frac{f_{z,C}}{\Theta_{i}}\left({d_{i}}-\theta_{i3}{d_{3}}\right). |   
  
This formula is the main result of the current note. In a similar way one can write formulas for jumps of slow variables due to other passages between domains Gjsubscript𝐺𝑗G_{j} that occur for other signs of values Θj,j=1,2,3formulae-sequencesubscriptΘ𝑗𝑗123\Theta_{j},j=1,2,3.

Value ξisubscript𝜉𝑖\xi_{i} is called a crossing parameter or a pseudo-phase. Asymptotic formulas for the pseudo-phase were obtained in [6] for Hamiltonian systems with one degree of freedom and slow time dependence, in [9] for slow-fast Hamiltonian systems with one degree of freedom corresponding to fast motion, in [3, 4] for motion in a slowly time-dependent potential with a dissipation, and in [11] for a general perturbed system of form (1.1).

Remark. We do not indicate accuracy of formula (LABEL:z_3i). One can see that terms ∼ε/ln⁡hNsimilar-toabsent𝜀subscriptℎ𝑁\sim\varepsilon/\ln h_{N} are neglected in some intermediate relations. However, because the final result should not depend on choice of hNsubscriptℎ𝑁h_{N}, the accuracy of the final formula should be much better. For Hamiltonian perturbations the accuracy of the final formula is O​(ε3/2​(|ln⁡ε|+(1−ξi)−1))𝑂superscript𝜀32𝜀superscript1subscript𝜉𝑖1O(\varepsilon^{3/2}(|\ln\varepsilon|+(1-\xi_{i})^{-1})) [7, 8].

###  3.6 Shift of slow time

The slow time τ𝜏\tau can be considered as a particular slow variable, τ˙=ε˙𝜏𝜀\dot{\tau}=\varepsilon. The formula for jump (or sift) of slow time for passage from G3subscript𝐺3G_{3} to Gisubscript𝐺𝑖G_{i}, i=1​or​ 2𝑖1or2i=1\ {\rm or}\ 2, is a particular case of (LABEL:z_3i) with fz,C=1subscript𝑓𝑧𝐶1f_{z,C}=1, Aj=0,j=1,2,3formulae-sequencesubscript𝐴𝑗0𝑗123A_{j}=0,j=1,2,3. Thus we get

|  | τ^i,∗−τ^3,∗≃ε​a​(ξi−12)​(ln⁡(ε​Θi)−2​θi​3​ln⁡(ε​Θ3))similar-to-or-equalssubscript^𝜏𝑖subscript^𝜏3𝜀𝑎subscript𝜉𝑖12𝜀subscriptΘ𝑖2subscript𝜃𝑖3𝜀subscriptΘ3\displaystyle\hat{\tau}_{i,*}-\hat{\tau}_{3,*}\simeq\varepsilon a(\xi_{i}-\frac{1}{2})(\ln(\varepsilon\Theta_{i})-2\theta_{i3}\ln(\varepsilon\Theta_{3})) |  | (3.14)  
---|---|---|---|---  
|  | −ε​a​ln⁡(2​π)3/2Γ(ξi)Γ(θi​3(1−ξi)Γ(1−θi​3ξi)\displaystyle-\varepsilon a\ln{\frac{(2\pi)^{3/2}}{\Gamma(\xi_{i})\Gamma(\theta_{i3}(1-\xi_{i})\Gamma(1-\theta_{i3}\xi_{i})}} |   
|  | −ε​(ξi−12)​(bi−θi​3​b3)+εΘi​(di−θi​3​d3).𝜀subscript𝜉𝑖12subscript𝑏𝑖subscript𝜃𝑖3subscript𝑏3𝜀subscriptΘ𝑖subscript𝑑𝑖subscript𝜃𝑖3subscript𝑑3\displaystyle-\varepsilon(\xi_{i}-\frac{1}{2})(b_{i}-\theta_{i3}b_{3})+\frac{\varepsilon}{\Theta_{i}}\left({d_{i}}-\theta_{i3}{d_{3}}\right). |   
  
##  4 Jump of adiabatic invariant

In this Section we derive formulas for jumps of adiabatic invariants in Hamiltonian systems from obtained formulas for jumps of slow variables.

###  4.1 Time-dependent Hamiltonian system

Let system (1.1) be a Hamiltonian system with the Hamiltonian H=H(p,q.τ)H=H(p,q.\tau), τ=ε​t𝜏𝜀𝑡\tau=\varepsilon t. Denote Si​(τ)subscript𝑆𝑖𝜏S_{i}(\tau) area of the domain Gisubscript𝐺𝑖G_{i}, i=1,2𝑖12i=1,2. Denote S3​(τ)=S1​(τ)∪S2​(τ)subscript𝑆3𝜏subscript𝑆1𝜏subscript𝑆2𝜏S_{3}(\tau)=S_{1}(\tau)\cup S_{2}(\tau). Then Θj=d​Sj/d​τsubscriptΘ𝑗𝑑subscript𝑆𝑗𝑑𝜏\Theta_{j}=dS_{j}/d\tau, j=1,2,3𝑗123j=1,2,3. Consider motion with passage from G3subscript𝐺3G_{3} to Gisubscript𝐺𝑖G_{i}, i=1​or​ 2𝑖1or2i=1\ {\rm or}\ 2, as in Section 3. Let J−subscript𝐽J_{-} and J+subscript𝐽J_{+} be the initial (at t=0𝑡0t=0, in G3subscript𝐺3G_{3}) and final (at t=K/ε𝑡𝐾𝜀t=K/\varepsilon, in Gisubscript𝐺𝑖G_{i}) values of the improved adiabatic invariant. (For the definition of the improved adiabatic invariant and related formulas see, e.g., [7]). Then S3​(τ^3,∗)≃2​π​J−similar-to-or-equalssubscript𝑆3subscript^𝜏32𝜋subscript𝐽S_{3}(\hat{\tau}_{3,*})\simeq 2\pi J_{-}, Si​(τ^i,∗)≃2​π​J+similar-to-or-equalssubscript𝑆𝑖subscript^𝜏𝑖2𝜋subscript𝐽S_{i}(\hat{\tau}_{i,*})\simeq 2\pi J_{+}. We get

| 2​π​J+≃Si​(τ^i,∗)=Si​(τ^3,∗+τ^i,∗−τ^3,∗)≃Si​(τ^3,∗)+Θi​(τ^i,∗−τ^3,∗).similar-to-or-equals2𝜋subscript𝐽subscript𝑆𝑖subscript^𝜏𝑖subscript𝑆𝑖subscript^𝜏3subscript^𝜏𝑖subscript^𝜏3similar-to-or-equalssubscript𝑆𝑖subscript^𝜏3subscriptΘ𝑖subscript^𝜏𝑖subscript^𝜏3\displaystyle 2\pi J_{+}\simeq S_{i}(\hat{\tau}_{i,*})=S_{i}(\hat{\tau}_{3,*}+\hat{\tau}_{i,*}-\hat{\tau}_{3,*})\simeq S_{i}(\hat{\tau}_{3,*})+\Theta_{i}(\hat{\tau}_{i,*}-\hat{\tau}_{3,*}). |  | (4.1)  
---|---|---|---  
  
Substitute (τ^i,∗−τ^3,∗)subscript^𝜏𝑖subscript^𝜏3(\hat{\tau}_{i,*}-\hat{\tau}_{3,*}) from (LABEL:tau_3i). We get

|  | 2​π​J+≃Si​(τ^3,∗)+ε​a​Θi​(ξi−12)​(ln⁡(ε​Θi)−2​θi​3​ln⁡(ε​Θ3))similar-to-or-equals2𝜋subscript𝐽subscript𝑆𝑖subscript^𝜏3𝜀𝑎subscriptΘ𝑖subscript𝜉𝑖12𝜀subscriptΘ𝑖2subscript𝜃𝑖3𝜀subscriptΘ3\displaystyle 2\pi J_{+}\simeq S_{i}(\hat{\tau}_{3,*})+\varepsilon a\Theta_{i}(\xi_{i}-\frac{1}{2})(\ln(\varepsilon\Theta_{i})-2\theta_{i3}\ln(\varepsilon\Theta_{3})) |  | (4.2)  
---|---|---|---|---  
|  | −ε​a​Θi​ln⁡(2​π)3/2Γ(ξi)Γ(θi​3(1−ξi)Γ(1−θi​3ξi)\displaystyle-\varepsilon a\Theta_{i}\ln{\frac{(2\pi)^{3/2}}{\Gamma(\xi_{i})\Gamma(\theta_{i3}(1-\xi_{i})\Gamma(1-\theta_{i3}\xi_{i})}} |   
|  | −Θi​(ξi−12)​(bi−θi​3​b3)+ε​(di−θi​3​d3)subscriptΘ𝑖subscript𝜉𝑖12subscript𝑏𝑖subscript𝜃𝑖3subscript𝑏3𝜀subscript𝑑𝑖subscript𝜃𝑖3subscript𝑑3\displaystyle-\Theta_{i}(\xi_{i}-\frac{1}{2})(b_{i}-\theta_{i3}b_{3})+{\varepsilon}\left({d_{i}}-\theta_{i3}{d_{3}}\right) |   
  
as in [5, 7]. One can replace Si​(τ^3,∗)subscript𝑆𝑖subscript^𝜏3S_{i}(\hat{\tau}_{3,*}) with Si​(τ∗)+θi​3​(2​π​J−−S3​(τ∗))subscript𝑆𝑖subscript𝜏subscript𝜃𝑖32𝜋subscript𝐽subscript𝑆3subscript𝜏S_{i}(\tau_{*})+\theta_{i3}(2\pi J_{-}-S_{3}(\tau_{*})) here.

###  4.2 Slow-fast Hamiltonian system

Let system (1.1) be a slow-fast Hamiltonian system. The Hamiltonian is H​(p,q,y,x)𝐻𝑝𝑞𝑦𝑥H(p,q,y,x) with pairs of conjugate variables (p,q)𝑝𝑞(p,q) and (y,ε−1​x)𝑦superscript𝜀1𝑥(y,\varepsilon^{-1}x). Equations of motion are

| q˙=∂H∂p,p˙=−∂H∂q,x˙=ε​∂H∂y,y˙=−ε​∂H∂x.formulae-sequence˙𝑞𝐻𝑝formulae-sequence˙𝑝𝐻𝑞formulae-sequence˙𝑥𝜀𝐻𝑦˙𝑦𝜀𝐻𝑥\displaystyle\dot{q}=\frac{\partial H}{\partial p},\ \dot{p}=-\frac{\partial H}{\partial q},\ \dot{x}=\varepsilon\frac{\partial H}{\partial y},\ \dot{y}=-\varepsilon\frac{\partial H}{\partial x}. |  | (4.3)  
---|---|---|---  
  
Thus, z=(y,x)𝑧𝑦𝑥z=(y,x), fz,C=(−∂hC​(y,x)/∂x,∂hC​(y,x)/∂x)subscript𝑓𝑧𝐶subscriptℎ𝐶𝑦𝑥𝑥subscriptℎ𝐶𝑦𝑥𝑥f_{z,C}=(-{\partial h_{C}(y,x)}/{\partial x},{\partial h_{C}(y,x)}/{\partial x}). Denote Si​(z)=Si​(y,x)subscript𝑆𝑖𝑧subscript𝑆𝑖𝑦𝑥S_{i}(z)=S_{i}(y,x) area of domain Gisubscript𝐺𝑖G_{i}, i=1,2𝑖12i=1,2. Denote S3​(z)=S1​(z)∪S2​(z)subscript𝑆3𝑧subscript𝑆1𝑧subscript𝑆2𝑧S_{3}(z)=S_{1}(z)\cup S_{2}(z). Then Θj={Sj,hc},j=1,2,3formulae-sequencesubscriptΘ𝑗subscript𝑆𝑗subscriptℎ𝑐𝑗123\Theta_{j}=\\{S_{j},h_{c}\\},j=1,2,3, where {⋅,⋅}⋅⋅\\{\cdot,\cdot\\} is the Poisson bracket with respect to variables (y,x)𝑦𝑥(y,x), {a,b}=ax′​by′−ay′​bx′𝑎𝑏subscriptsuperscript𝑎′𝑥subscriptsuperscript𝑏′𝑦subscriptsuperscript𝑎′𝑦subscriptsuperscript𝑏′𝑥\\{a,b\\}=a^{\prime}_{x}b^{\prime}_{y}-a^{\prime}_{y}b^{\prime}_{x} (see [8]) and

| Aj=(−∮lj∂(H−hc)∂x​𝑑t,∮lj∂(H−hc)∂y​𝑑t)=(∂Sj∂x,−∂Sj∂y).subscript𝐴𝑗subscriptcontour-integralsubscript𝑙𝑗𝐻subscriptℎ𝑐𝑥differential-d𝑡subscriptcontour-integralsubscript𝑙𝑗𝐻subscriptℎ𝑐𝑦differential-d𝑡subscript𝑆𝑗𝑥subscript𝑆𝑗𝑦A_{j}=\left(-\oint_{l_{j}}\frac{\partial(H-h_{c})}{\partial x}dt,\oint_{l_{j}}\frac{\partial(H-h_{c})}{\partial y}dt\right)=\left(\frac{\partial S_{j}}{\partial x},-\frac{\partial S_{j}}{\partial y}\right). |  | (4.4)  
---|---|---|---  
  
(cf. [8]).

Consider motion with passage from G3subscript𝐺3G_{3} to Gisubscript𝐺𝑖G_{i}, i=1​or​ 2𝑖1or2i=1\ {\rm or}\ 2, as in Section 3. Let J−subscript𝐽J_{-} and J+subscript𝐽J_{+} be the initial (at t=0𝑡0t=0, in G3subscript𝐺3G_{3}) and final (at t=K/ε𝑡𝐾𝜀t=K/\varepsilon, in Gisubscript𝐺𝑖G_{i}) values of the improved adiabatic invariant. (For the definition of the improved adiabatic invariant and related formulas see, e.g., [8]). Then S3​(z^3,∗)≃2​π​J−similar-to-or-equalssubscript𝑆3subscript^𝑧32𝜋subscript𝐽S_{3}(\hat{z}_{3,*})\simeq 2\pi J_{-}, Si​(z^i,∗)≃2​π​J+similar-to-or-equalssubscript𝑆𝑖subscript^𝑧𝑖2𝜋subscript𝐽S_{i}(\hat{z}_{i,*})\simeq 2\pi J_{+}. Then we get

| 2​π​J+≃Si​(z^i,∗)=Si​(z^3,∗+z^i,∗−z^3,∗)≃Si​(z^3,∗)+(grad​Si⋅(z^i,∗−z^3,∗)).similar-to-or-equals2𝜋subscript𝐽subscript𝑆𝑖subscript^𝑧𝑖subscript𝑆𝑖subscript^𝑧3subscript^𝑧𝑖subscript^𝑧3similar-to-or-equalssubscript𝑆𝑖subscript^𝑧3⋅gradsubscript𝑆𝑖subscript^𝑧𝑖subscript^𝑧3\displaystyle 2\pi J_{+}\simeq S_{i}(\hat{z}_{i,*})=S_{i}(\hat{z}_{3,*}+\hat{z}_{i,*}-\hat{z}_{3,*})\simeq S_{i}(\hat{z}_{3,*})+({\rm grad}\,S_{i}\cdot(\hat{z}_{i,*}-\hat{z}_{3,*})). |  | (4.5)  
---|---|---|---  
  
Here (⋅)⋅(\phantom{*}\cdot\phantom{*}) is the standard scalar product. Substitute (z^i,∗−z^3,∗)subscript^𝑧𝑖subscript^𝑧3(\hat{z}_{i,*}-\hat{z}_{3,*}) from (LABEL:z_3i) and note that

| (grad​Si⋅fz,C)=Θi,(grad​Si⋅Ai)={Si,Si}=0,(grad​Si⋅A3)=−{Si,S3}.formulae-sequenceformulae-sequence⋅gradsubscript𝑆𝑖subscript𝑓𝑧𝐶subscriptΘ𝑖⋅gradsubscript𝑆𝑖subscript𝐴𝑖subscript𝑆𝑖subscript𝑆𝑖0⋅gradsubscript𝑆𝑖subscript𝐴3subscript𝑆𝑖subscript𝑆3\displaystyle({\rm grad}\,S_{i}\cdot f_{z,C})=\Theta_{i},\ ({\rm grad}\,S_{i}\cdot A_{i})=\\{S_{i},S_{i}\\}=0,\ ({\rm grad}\,S_{i}\cdot A_{3})=-\\{S_{i},S_{3}\\}. |  | (4.6)  
---|---|---|---  
  
We get

|  | 2​π​J+≃Si​(z^3,∗)+ε​Θi​a​(ξi−12)​(ln⁡(ε​Θi)−2​θi​3​ln⁡(ε​Θ3))similar-to-or-equals2𝜋subscript𝐽subscript𝑆𝑖subscript^𝑧3𝜀subscriptΘ𝑖𝑎subscript𝜉𝑖12𝜀subscriptΘ𝑖2subscript𝜃𝑖3𝜀subscriptΘ3\displaystyle 2\pi J_{+}\simeq S_{i}(\hat{z}_{3,*})+\varepsilon\Theta_{i}a(\xi_{i}-\frac{1}{2})(\ln(\varepsilon\Theta_{i})-2\theta_{i3}\ln(\varepsilon\Theta_{3})) |  | (4.7)  
---|---|---|---|---  
|  | −ε​a​Θi​ln⁡(2​π)3/2Γ(ξi)Γ(θi​3(1−ξi)Γ(1−θi​3ξi)\displaystyle-\varepsilon a\Theta_{i}\ln{\frac{(2\pi)^{3/2}}{\Gamma(\xi_{i})\Gamma(\theta_{i3}(1-\xi_{i})\Gamma(1-\theta_{i3}\xi_{i})}} |   
|  | −ε​Θi​(ξi−12)​(bi−θi​3​b3)−ε​θi​3​(ξi−12)​{Si,S3}𝜀subscriptΘ𝑖subscript𝜉𝑖12subscript𝑏𝑖subscript𝜃𝑖3subscript𝑏3𝜀subscript𝜃𝑖3subscript𝜉𝑖12subscript𝑆𝑖subscript𝑆3\displaystyle-\varepsilon\Theta_{i}(\xi_{i}-\frac{1}{2})(b_{i}-\theta_{i3}b_{3})-\varepsilon\theta_{i3}(\xi_{i}-\frac{1}{2})\\{S_{i},S_{3}\\} |   
|  | +ε​(di−θi​3​d3).𝜀subscript𝑑𝑖subscript𝜃𝑖3subscript𝑑3\displaystyle+\varepsilon\left({d_{i}}-\theta_{i3}{d_{3}}\right). |   
  
For systems with two degrees of freedom one can approximately calculate Si​(z^3,∗)subscript𝑆𝑖subscript^𝑧3S_{i}(\hat{z}_{3,*}) via initial value of the improved adiabatic invariant and solution of the first order averaged system. Consider motion in the energy level H=h𝐻ℎH=h. Relations

| S3​(z^3,∗)≃2​π​J−,hC​(z^3,∗)≃h,hC​(z∗)=hformulae-sequencesimilar-to-or-equalssubscript𝑆3subscript^𝑧32𝜋subscript𝐽formulae-sequencesimilar-to-or-equalssubscriptℎ𝐶subscript^𝑧3ℎsubscriptℎ𝐶subscript𝑧ℎS_{3}(\hat{z}_{3,*})\simeq 2\pi J_{-},\quad h_{C}(\hat{z}_{3,*})\simeq h,\quad h_{C}(z_{*})=h |  | (4.8)  
---|---|---|---  
  
imply that

| (grad​S3⋅(z^3,∗−z∗))≃2​π​J−−S3​(z∗),(grad​hC⋅(z^3,∗−z∗))≃0.formulae-sequencesimilar-to-or-equals⋅gradsubscript𝑆3subscript^𝑧3subscript𝑧2𝜋subscript𝐽subscript𝑆3subscript𝑧similar-to-or-equals⋅gradsubscriptℎ𝐶subscript^𝑧3subscript𝑧0\displaystyle({\rm grad}\,S_{3}\cdot(\hat{z}_{3,*}-z_{*}))\simeq 2\pi J_{-}-S_{3}(z_{*}),\quad({\rm grad}\,h_{C}\cdot(\hat{z}_{3,*}-z_{*}))\simeq 0. |  | (4.9)  
---|---|---|---  
  
We have

| Si​(z^3,∗)=Si​(z∗+z^3,∗−z∗)≃Si​(z∗)+(grad​Si⋅(z^3,∗−z∗)).subscript𝑆𝑖subscript^𝑧3subscript𝑆𝑖subscript𝑧subscript^𝑧3subscript𝑧similar-to-or-equalssubscript𝑆𝑖subscript𝑧⋅gradsubscript𝑆𝑖subscript^𝑧3subscript𝑧S_{i}(\hat{z}_{3,*})=S_{i}(z_{*}+\hat{z}_{3,*}-z_{*})\simeq S_{i}(z_{*})+({\rm grad}\,S_{i}\cdot(\hat{z}_{3,*}-z_{*})). |  | (4.10)  
---|---|---|---  
  
Solve equations (4.9) for (z^3,∗−z∗)subscript^𝑧3subscript𝑧(\hat{z}_{3,*}-z_{*}) and substitute the result to (4.10). We get

| Si​(z^3,∗)≃Si​(z∗)+θi​3​(2​π​J−−S3​(z∗)).similar-to-or-equalssubscript𝑆𝑖subscript^𝑧3subscript𝑆𝑖subscript𝑧subscript𝜃𝑖32𝜋subscript𝐽subscript𝑆3subscript𝑧S_{i}(\hat{z}_{3,*})\simeq S_{i}(z_{*})+\theta_{i3}(2\pi J_{-}-S_{3}(z_{*})). |  | (4.11)  
---|---|---|---  
  
Substitution of this relation to (LABEL:J_sf) gives an expression for jump of the adiabatic invariant in [8].

##  5 Conclusions

The main result of this note is the asymptotic formula (LABEL:z_3i) for change of slow variables at evolution across separatrices in systems of form (1.1). Together with formula for phase change in such systems [11] this gives a rather complete description of dynamics with separatrix crossings in the considered class of systems.

## References

  * [1] Arnold V. I. Mathematical Methods of Classical Mechanics: Graduate Texts in Mathematics 60. Springer-Verlag, New York (1978), x+462 pp. 
  * [2] Bogolyubov N. N., Mitropolskij Yu. A. Asymptotic Methods in the Theory of Non-Linear Oscillations. Hindustan Publ. Corp., Delhi; Gordon and Breach Sci. Publ., New York (1961), x+537 pp. 
  * [3] Bourland F. J., Haberman R. Separatrix crossing: time-invariant potentials with dissipation. SIAM J. Appl. Math., 50, 6, 1716–1744 (1990) 
  * [4] Bourland F. J., Haberman R. Connection across a separatrix with dissipation. Stud. Appl. Math., 91, 95–124 (1994) 
  * [5] Cary J. R., Escande D. F., Tennyson J. L.  Adiabatic-invariant change due to separatrix crossing.  Physical Review A, 34, 4256–4275 (1986) 
  * [6] Cary J. R., Skodje R. T. Phase change between separatrix crossings. Physica D, 36, 3, 287–316 (1989) 
  * [7] Neishtadt A. I.  Change of an adiabatic invariant at a separatrix.  Soviet Journal of Plasma Physics, 12, 568–573 (1986) 
  * [8] Neishtadt A. I. On the change in the adiabatic invariant on crossing a separatrix in systems with two degrees of freedom. J. Appl. Math. Mech., 51, 5, 586–592 (1987) 
  * [9] Neishtadt A. I., Vasiliev A. A. Phase change between separatrix crossings in slow-fast Hamiltonian systems.Nonlinearity, 18, 3, 1393–1406 (2005) 
  * [10] Neishtadt A. I. Averaging method for systems with separatrix crossing. Nonlinearity, 30, 7, 2871–2917 (2017) 
  * [11] Neishtadt A. I., Okunev A.V. Phase change and order 2 averaging for one-frequency systems with separatrix crossing. Nonlinearity, 35, 8, 4469–4516 (2022) 
  * [12] Timofeev A. V. On the constancy of an adiabatic invariant when the nature of the motion changes. Sov. Phys., JETP, 48, 656–659 (1978) 

  

Anatoly Neishtadt

Department of Mathematical Sciences

Loughborough University, Loughborough LE11 3TU, United Kingdom

E-mail: a.neishtadt@lboro.ac.uk

[◄](/html/2306.05764) [](/) [Feeling  
lucky?](/feeling_lucky) [](/land_of_honey_and_milk) [Conversion  
report](/log/2306.05765) [Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2306.05765) [View original  
on arXiv](https://arxiv.org/abs/2306.05765)[►](/html/2306.05766)

[](javascript:toggleColorScheme\(\) "Toggle ar5iv color scheme") [Copyright](https://arxiv.org/help/license) [Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Feb 29 02:07:23 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)
