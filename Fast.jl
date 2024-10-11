include("logw.jl")
include("lapl.jl")
include("Approx.jl")
include("Exact.jl")
include("graph.jl")
include("bb.jl")
import Dates

rightnow = Dates.now()
w= open("E:\\fairness\\output\\log.txt", "a")
logw(w, "\n")
logw(w, rightnow)
close(w)
#configuration
for kkkk in [0.0, 0.2, 0.5, 0.8, 1.0]
lambda = kkkk
com = true
fname = open("E:\\fairness\\code\\filename.txt", "r")
str = readline(fname);
nn = parse(Int, str);

for iii = 1:nn 
str = readline(fname);
str = strip(str);
println(str)
w= open("E:\\fairness\\output\\log.txt", "a")
logw(w, str)
close(w)
G = get_graph(str,com)

n = G.n
m = G.m
S_rto = length(G.mem0)/n
T_rto = length(G.mem1)/n
S_num = length(G.mem0)
T_num = length(G.mem1)
L = lapsp(G);
A = adjsp(G);

w= open(string("E:\\fairness\\output\\",G.name,"_applog.txt"), "a")
logw(w, lambda)
close(w)
rst_idx = [0 0];
for e=1:50
    appDiag = LinvdiagSS(A;JLfac=100)
    appInvTrace = sum(appDiag)
    w= open(string("E:\\fairness\\output\\",G.name,"_applog50.txt"), "a")
    ini_disparity = (1/S_rto*sum(appDiag[G.mem0])) - (1/T_rto*sum(appDiag[G.mem1]))
    if e != 1
        logw(w, rst_idx[1], " ",rst_idx[2], " ", appInvTrace, " ", ini_disparity)
    end
    close(w)

    obj = (1 - lambda) * appInvTrace + lambda * ((1/S_rto*sum(appDiag[G.mem0])+appInvTrace)^2
                                                +(1/T_rto*sum(appDiag[G.mem1])+appInvTrace)^2)


    den1 = appInvTrace; den2 = (1/S_rto*sum(appDiag[G.mem0])+appInvTrace)^2 + (1/T_rto*sum(appDiag[G.mem1])+appInvTrace)^2;  

    tt1 = 1/S_rto*sum(appDiag[G.mem0])
    tt2 = 1/T_rto*sum(appDiag[G.mem1])
    xishu1 = (1 - lambda)/den1 + 2 * lambda * (appInvTrace + appInvTrace + tt1 + tt2)/den2
    xishu2 = (2 * lambda * 1/S_rto) * (appInvTrace + tt1)/den2
    xishu3 = (2 * lambda * 1/T_rto) * (appInvTrace + tt2)/den2




    f = approxchol_lap(A);
    JLfac = 100;
    k = round(Int, JLfac*log(n))

    xx = zeros(n,3*k)
    E0 = zeros(n);
    E1 = zeros(n);
    E0[G.mem0] .= 1;
    E1[G.mem1] .= 1;
    rr = zeros(n);
    v = zeros(n)
    for i = 1:k
        r = randn(n,1)
        v .= f(r[:])
        xx[:,i] .= sqrt(xishu1) * v

        rr .= E0 .* r[:]
        v .= f(rr[:])
        xx[:,k+i] .= sqrt(xishu2) * v

        rr .= E1 .* r[:]
        v .= f(rr[:])
        xx[:,2*k+i] .= sqrt(xishu3) * v
    end

    rst_nd = bb(xx)
    ei = zeros(n);
    ej = zeros(n);
    rst = 0;

    z = zeros(n,1)
    for i = 1:length(rst_nd)
        ei[rst_nd[i]] = 1;
        for j = i+1:length(rst_nd)
            ej[rst_nd[j]] = 1;
            be = ei - ej;
            z .= f(be[:])
            c1 = xishu2 * sum(z[G.mem0].^2)
            c2 = xishu3 * sum(z[G.mem1].^2)
            c3 = xishu1 * sum(z.^2)
            t = c1 + c2 + c3;
            # global rst
            if t > rst
                rst = t;
                rst_idx[1] = rst_nd[i];
                rst_idx[2] = rst_nd[j];
            end
            ej[rst_nd[j]] = 0;
        end
        ei[rst_nd[i]] = 0;
    end
    xx = rst_idx[1]; yy = rst_idx[2];
    L[xx, xx] += 1; L[yy, yy] += 1; L[xx, yy] = -1; L[yy, xx] = -1;
    A[xx, yy] = 1; A[yy, xx] = 1;

end
appDiag = LinvdiagSS(A;JLfac=100)
appInvTrace = sum(appDiag)
w= open(string("E:\\fairness\\output\\",G.name,"_applog50.txt"), "a")
ini_disparity = 1/S_rto*sum(appDiag[G.mem0]) - 1/T_rto*sum(appDiag[G.mem1])

logw(w, rst_idx[1], " ",rst_idx[2], " ", appInvTrace, " ", ini_disparity)
close(w)

end
end