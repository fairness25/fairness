include("graph.jl")
include("lapl.jl")
include("logw.jl")
function exact_aft_deri(G, lambda)
    out_path = "../output"
    n = G.n; m = G.m;
    S_rto = length(G.mem0)/n
    T_rto = length(G.mem1)/n
    L = lapsp(G);
    A = adjsp(G);


    Linv = mppinv(L)
    InvTrace = 0;
    InvDiag = zeros(n);
    for i = 1:n
        InvTrace = InvTrace + Linv[i,i]
        InvDiag[i] = Linv[i,i]
    end

    den1 = InvTrace; den2 = (1/S_rto*sum(InvDiag[G.mem0])+InvTrace)^2 + (1/T_rto*sum(InvDiag[G.mem1])+InvTrace)^2;  

    edges = Dict{Tuple{Int, Int}, Int}()
    for i =1:G.m
        ii = G.u[i]
        jj = G.v[i]
        if !haskey(edges, (ii, jj)) && !haskey(edges, (jj, ii))
            edges[(ii, jj)] = 1
        end
    end
    c_edges = Dict{Tuple{Int, Int}, Int}()
    for i = 1:G.n
        for j = i+1:G.n
            if A[i,j]==0
                c_edges[(i, j)] = 1
            end
        end
    end

    ei = zeros(n);
    ej = zeros(n);
    InvTrace = 0;
    InvDiag = zeros(n);
    for kk = 1:50
        w = open(string("E:\\fairness\\output\\", G.name, "_aftlog",lambda, ".txt"), "a")
        Linv = mppinv(L)
        InvTrace = 0;
        InvDiag = zeros(n);
        for i = 1:n
            InvTrace = InvTrace + Linv[i,i]
            InvDiag[i] = Linv[i,i]
        end
        tt1 = 1/S_rto*sum(InvDiag[G.mem0])
        tt2 = 1/T_rto*sum(InvDiag[G.mem1])
        xishu1 = (1 - lambda)/den1 + 2 * lambda * (InvTrace + InvTrace + tt1 + tt2)/den2#ld*ld， 1，4，5
        xishu2 = (2 * lambda * 1/S_rto) * (InvTrace + tt1)/den2#ldevevld, v in S， 2
        xishu3 = (2 * lambda * 1/T_rto) * (InvTrace + tt2)/den2#ldevevld, v in T， 3
        
        rst = 0;
        rst_idx = [0 0];
        iii = 0
        z = zeros(n)
        for (i, j) in keys(c_edges)
            if iii % 100000 == 0
                println(iii)
            end
            iii += 1
            z .= Linv[:,i] - Linv[:,j]
            c1 = xishu2 * sum(z[G.mem0].^2)
            c2 = xishu3 * sum(z[G.mem1].^2)
            c3 = xishu1 * sum(z.^2)
            t = c1 + c2 + c3;
            if t > rst
                rst = t;
                rst_idx[1] = i;
                rst_idx[2] = j;
            end
            # ei[i] = 0; ej[j] = 0;
        end
        xx = rst_idx[1]; yy = rst_idx[2];
        delete!(c_edges, (xx, yy))
        L[xx, xx] += 1; L[yy, yy] += 1; L[xx, yy] = -1; L[yy, xx] = -1;
        
        Linv = mppinv(L)
        InvTrace = 0;
        for i = 1:n
            InvTrace = InvTrace + Linv[i,i]
            InvDiag[i] = Linv[i,i]
        end
        disparity = 1/S_rto*sum(InvDiag[G.mem0]) - 1/T_rto*sum(InvDiag[G.mem1])
        kir = InvTrace;
        logw(w, rst_idx[1], ", ", rst_idx[2], ", ", kir, ", ", disparity)
        close(w)
    end
    # close(w)
    # return rst, rst_idx
end

function exact_bf_deri(G, lambda)
    out_path = "E:\\fairness\\output\\"
    ww = open(string(out_path, "logg.txt"), "w")
    n = G.n
    S_rto = length(G.mem0)/n
    T_rto = length(G.mem1)/n
    L = lapsp(G);
    A = adjsp(G);

    edges = Dict{Tuple{Int, Int}, Int}()
    for i =1:G.m
        ii = G.u[i]
        jj = G.v[i]
        if !haskey(edges, (ii, jj)) && !haskey(edges, (jj, ii))
            edges[(ii, jj)] = 1
        end
    end
    
    c_edges = Dict{Tuple{Int, Int}, Int}()
    for i = 1:G.n
        for j = i+1:G.n
            if A[i,j]==0
                c_edges[(i, j)] = 1
            end
        end
    end
    
    Linv = mppinv(L)
    InvTrace = 0;
    InvDiag = zeros(n);
    for i = 1:n
        InvTrace = InvTrace + Linv[i,i]
        InvDiag[i] = Linv[i,i]
    end
    den1 = InvTrace; den2 = (1/S_rto*sum(InvDiag[G.mem0])+InvTrace)^2 + (1/T_rto*sum(InvDiag[G.mem1])+InvTrace)^2;  
    for k = 1:50
        w = open(string("E:\\fairness\\output\\", G.name, "_bflog", lambda, ".txt"), "a")
        Linv = mppinv(L)
        InvTrace = 0;
        InvDiag = zeros(n);
        for i = 1:n
            InvTrace = InvTrace + Linv[i,i]
            InvDiag[i] = Linv[i,i]
        end
        ini_obj = (1 - lambda) * InvTrace/den1 + lambda * ((1/S_rto*sum(InvDiag[G.mem0])+InvTrace)^2
                                                    +(1/T_rto*sum(InvDiag[G.mem1])+InvTrace)^2)/den2

        ini_disparity = 1/S_rto*sum(InvDiag[G.mem0]) - 1/T_rto*sum(InvDiag[G.mem1])
        ini_kir = InvTrace;
        rst = ini_obj;
        rst_idx = [0 0];
        rst_kir = ini_kir;
        rst_dis = ini_disparity;

        c = zeros(n,1)
        InvDiag = zeros(n,1);
        B = zeros(n,1);
        Ld_diag = zeros(n,1);
        iii = 0
        for i = 1:n
            Ld_diag[i] = Linv[i,i]
        end
        for (i, j) in keys(c_edges)
            if iii % 100000 == 0
                println(iii)
            end
            iii += 1
            c .= Linv[:,i] - Linv[:,j]
            B .= c.^2
            B ./= (1 + Linv[i,i] + Linv[j,j] - Linv[i,j] - Linv[j,i])
            InvDiag .= (Ld_diag .- B)
            
            InvTrace = sum(InvDiag)
            obj = (1 - lambda) * InvTrace/den1 + lambda * ((1/S_rto*sum(InvDiag[G.mem0])+InvTrace)^2
                                                        +(1/T_rto*sum(InvDiag[G.mem1])+InvTrace)^2)/den2

            disparity = 1/S_rto*sum(InvDiag[G.mem0]) - 1/T_rto*sum(InvDiag[G.mem1])
            if obj < rst
                rst = obj
                rst_dis = disparity
                rst_kir = InvTrace
                rst_idx[1] = i
                rst_idx[2] = j
            end

        end
        # close(fout)
        xx = rst_idx[1]; yy = rst_idx[2];
        delete!(c_edges, (xx, yy))
        L[xx, xx] += 1; L[yy, yy] += 1; L[xx, yy] = -1; L[yy, xx] = -1;
        println(ww, "rst_idx:", rst_idx)
        logw(w, rst_idx[1], ", ", rst_idx[2], ", ", rst_kir, ", ", rst_dis)
        close(w)
    end
    close(ww)
end

com = true
fname = open("E:\\fairness\\code\\filename.txt", "r")
str = readline(fname);
nn = parse(Int, str);
for iii = 1:nn 
    str = readline(fname);
    str = strip(str);
    println(str)
    G = get_graph(str,com)
    for i in [0, 0.2, 0.5, 0.8, 1]
        exact_bf_deri(G, i)
        # exact_aft_deri(G, i)
    end
end
