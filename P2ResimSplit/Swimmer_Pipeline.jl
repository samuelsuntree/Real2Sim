using WaterLily
using ParametricBodies
using CSV,DataFrames
using Plots
using StaticArrays
using DelimitedFiles
using ReadVTK, WriteVTK
import WaterLily: scale_u!,BDIM!,project!,CFL,conv_diff!,BC!,project!,BCTuple


# 1. Parse command-line arguments
a1 = parse(Int, ARGS[1])  # video index
a2 = parse(Int, ARGS[2])  # crop index
a3 = parse(Float64, ARGS[3])  # externally passed "duration" (float)

# 2. Build path and read CSV
path = "C:/Users/10521/Documents/GitHub/Real2Sim/P1VideoCapture/output_files/$(a1)/$(a2)/final/"

x = CSV.read(path * "x.csv", DataFrame; header=false)
y = CSV.read(path * "y.csv", DataFrame; header=false)

# 3. Define scale and shift
scale = 1/5
x .= x .* (scale * 2000/(2^8))
y .= y .* (scale * 1200/(2^7))

x_bar_left = 655 * scale
y_bar_up   = (1200 - 845) * scale
x .= x .- x_bar_left
y .= y .- y_bar_up

# 4. Compute 'duration' internally as the frame count (size(x,2))
internal_duration = size(x, 2)/10

# 5. Compare internal_duration with externally passed a3
diff_value = abs(internal_duration - a3)
diff_percent = diff_value / a3 * 100  # relative difference in %

println("=======================================================================")
println("External a3 passed in:           ", a3)
println("Internal duration (size(x,2)):   ", internal_duration)
println("Absolute difference:             ", diff_value)
println("Relative difference:             ", round(diff_percent, digits=2), " %")
println("=======================================================================")

# 6. (Optional) If the difference is above some threshold, warn or continue
#    For example:
if diff_percent > 10
    @warn "Warning: The difference exceeds 10%!"
end

# 7. Downsample or further process x,y as usual, then continue..
x_ds = reverse(Array(x[1:15:end, :]), dims=1)
y_ds = reverse(Array(y[1:15:end, :]), dims=1)
# close the curve by adding a point between the first and last point (keeps continuitiy)
x_mid = 0.5sum(x_ds[[1,end],:],dims=1)
y_mid = 0.5sum(y_ds[[1,end],:],dims=1)
x_ds = vcat(x_mid,x_ds,x_mid)
y_ds = vcat(y_mid,y_ds,y_mid)
# interpolate in time to get smooth motion
using Interpolations
ξ = 1:20; t = 0:1/(size(x_ds,2)-1):1 # motion between 0 and 1
func_xs = interpolate((ξ,t), x_ds, Gridded(Linear()))
func_ys = interpolate((ξ,t), y_ds, Gridded(Linear()))
function fish_motion(t)
    return SMatrix{2,size(x_ds,1)}(hcat(func_xs(ξ,t),func_ys(ξ,t))')
end
@assert fish_motion(0) ≈ SMatrix{2,size(x_ds,1)}(hcat(x_ds[:,1],y_ds[:,1])')

#######viscous dimenstionalize based on same Re
#ν~/ν_water = dt_frame/t_step * (L~_cali_pixel/L_caliber)^2 
        #   = (0.01s/0.1)*(517/5/(14*0.019))^2 = 1.5110e+04
#ν~ = 1.5141e-02
#Re~ = Velo~_fish * L~ /ν~ ~= 400/5/2.6 * 50 / 1.5141e-02 = 101.61k (fast swimming)
#Re = Velo * L / ν = Re~
#######Important, please always check before running !!!!

# test plot
@gif for i ∈ 1:size(x,2)
    # x[!,i] = x[!,i] .+ 350*0
    # y[!,i] = y[!,i] .- 60*0
    
    xᵢ,yᵢ = x[!,i],y[!,i]
    
    plot(xᵢ,yᵢ, aspect_ratio=:equal)
    plot!(title="tU/L=$(i/10)")
end

function make_sim(L=32;Re=Float64(32/0.00095),U=1,T=Float64,mem=Array)
    cps = fish_motion(0)
    curve = BSplineCurve(cps;degree=2)
   
    # make a body
    body = DynamicNurbsBody(curve)

    # make sim
    Simulation((200,80),(0,0),L;U,ν=1.5141e-02,body,T=Float64,mem)
end





# run

base_dir = "Simu_result/"
VC_dir = joinpath(base_dir, "Video_$(a1)_Crop_$(a2)")

# 创建目录（包括子目录）如果它们不存在
mkpath(VC_dir)

# change to the output directory
cd(VC_dir) do 
    # make the sim
    sim = make_sim()
    t₀,duration,tstep = sim_time(sim),internal_duration,0.1;
    #duration = 3.5 # debug
    p,ν = [],[]
    moyFull = []
    # make a vtk writer
    wr = vtkWriter("SwimmerTest")
    print("Start simulation...",t₀)
    # plot the initial condition
    contourf(sim.flow.μ₀[:,:,1]',lw=0)
    println("Start simulation...")
    divu = zero(sim.flow.p)
    WaterLily.logger("fish_logger")


    #output_gif_path = joinpath("C:/Users/10521/Documents/GitHub/ParametricBodies.jl/example/", "swimmer5.gif")  # 自定义路径和文件名
    output_gif_path = "C:/Users/10521/Documents/GitHub/ParametricBodies.jl/example/"  # 自定义路径和文件名
    k = 1
    @gif for tᵢ in range(t₀,t₀+duration;step=tstep) 
        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])

        # every time step we do a momentum step
        while t < tᵢ*sim.L/sim.U
            new_pnts = fish_motion((t/sim.L/internal_duration)) # the time scaling is wired
            sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
            measure!(sim,t);
            # mom_step!(sim.flow,sim.pois) # evolve Flow
            # upack the momentum update here
            a = sim.flow; b=sim.pois; N=2
            a.u⁰ .= a.u; scale_u!(a,0); U = BCTuple(a.U,a.Δt,N)
            # predictor u → u'
            @log "p"
            conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν,perdir=a.perdir)
            BDIM!(a); BC!(a.u,U,a.exitBC,a.perdir)
            @inside divu[I] = WaterLily.div(I,a.u); # @TODO store to plot
            project!(a,b); BC!(a.u,U,a.exitBC,a.perdir)
            # corrector u → u¹
            @log "c"
            conv_diff!(a.f,a.u,a.σ,ν=a.ν,perdir=a.perdir)
            BDIM!(a); scale_u!(a,0.5); BC!(a.u,U,a.exitBC,a.perdir)
            project!(a,b,0.5); BC!(a.u,U,a.exitBC,a.perdir)
            push!(a.Δt,CFL(a))
            #
            t += sim.flow.Δt[end]
        end
        
        # plot for the gif
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
        contourf(clamp.(sim.flow.σ,-5,5)',dpi=300,
                color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
                aspect_ratio=:equal)#, legend=false, border=:none)
        #flood(sim.flow.σ; shift=(-0.5,-0.5),clims=(-5,5))
        plot!(sim.body.curve;shift=(1.5,1.5),add_cp=false)
        plot!(title="tU/L=$tᵢ")
        write!(wr, sim);
        
        #push!(p,WaterLily.pressure_force(sim)[1])
        #push!(ν,WaterLily.viscous_force(sim)[1])

        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        # open("Simu_result/HFSP/flow_p_$(round(tᵢ, digits=2)).txt", "w") do file
        #     writedlm(file, sim.flow.p |> Array)
        # end
    end #output_gif_path


    close(wr)

    #restart_sim = make_sim()

    #savefig("swimmer.gif")

    # plot(p,label="Pressure force")
    # plot!(ν,label="Viscous force")
end