import matplotlib.pyplot as plt
from Utils.general_utils import *

class PlotMaker:

    def __init__(self):
        return
    
    def make_plot(self, option):

        match option:
            case 1: # build graphs from data
                
                figname = "30_vs_res"
                titulo = "Architecture Comparison"
                plt.figure(figsize=(12, 6.8))

                data_path = "Graphs/_graph_data/solo_reduce_prog_4_1100_0-100-iterations_extrapolation.pkl"
                prog_win_rates = load_pickle(data_path)

                data_path = "Graphs/_graph_data/solo_res_5x5_to_10x10_win_rates.pkl"
                res_wr = load_pickle(data_path)

                for size_i in range(len(prog_win_rates)):
                    iterations = prog_win_rates[size_i]
                    line_lst = plt.plot(range(len(iterations)), iterations, label = str(size_i+5) + "x" + str(size_i+5))
                    color = line_lst[0].get_color()
                    wr = res_wr[size_i]
                    plt.axhline(y=wr, linestyle='--', label = str(size_i+5) + "x" + str(size_i+5) + "_ResNet", color = color)

                plt.xlabel("Recurrent Iterations")
                plt.ylabel("Win Ratio")
        
                plt.title(titulo, pad=20, fontsize = 14)

                lgd = plt.legend(bbox_to_anchor=(1,1))
                plt.gcf().canvas.draw()
                invFigure = plt.gcf().transFigure.inverted()

                lgd_pos = lgd.get_window_extent()
                lgd_coord = invFigure.transform(lgd_pos)
                lgd_xmax = lgd_coord[1, 0]

                ax_pos = plt.gca().get_window_extent()
                ax_coord = invFigure.transform(ax_pos)
                ax_xmax = ax_coord[1, 0]        

                shift = 1.1 - (lgd_xmax - ax_xmax)
                plt.gcf().tight_layout(rect=(0, 0, shift, 1))


                plt.savefig("Graphs/" + figname, dpi=300)
                plt.clf()

                return
            
            case 2: # build more graphs from data
                
                figname = "solo_extrapolation"
                titulo = "Solo Soldier Extrapolation"
                plt.figure(figsize=(12, 7))

                data_path = "Graphs/_graph_data/solo_final_1100_0-100-iterations.pkl"
                prog_win_rates = load_pickle(data_path)

                for size_i in range(len(prog_win_rates)):
                    iterations = prog_win_rates[size_i]
                    line_lst = plt.plot(range(len(iterations)), iterations, label = str(size_i+5) + "x" + str(size_i+5))
        

                plt.xlabel("Recurrent Iterations")
                plt.ylabel("Win Ratio")
        
                plt.title(titulo, pad=20, fontsize = 14)

                lgd = plt.legend(bbox_to_anchor=(1,1))
                plt.gcf().canvas.draw()
                invFigure = plt.gcf().transFigure.inverted()

                lgd_pos = lgd.get_window_extent()
                lgd_coord = invFigure.transform(lgd_pos)
                lgd_xmax = lgd_coord[1, 0]

                ax_pos = plt.gca().get_window_extent()
                ax_coord = invFigure.transform(ax_pos)
                ax_xmax = ax_coord[1, 0]        

                shift = 1.05 - (lgd_xmax - ax_xmax)
                plt.gcf().tight_layout(rect=(0, 0, shift, 1))


                plt.savefig("Graphs/" + figname, dpi=300)
                plt.clf()

                return
            
            case 3: # build even more graphs from data
                
                figname = "30_vs_res"
                titulo = "Architecture Comparison"
                plt.figure(figsize=(12, 6.8))

                data_path = "Graphs/_graph_data/solo_reduce_prog_4_1100_0-100-iterations_extrapolation.pkl"
                prog_win_rates = load_pickle(data_path)

                data_path = "Graphs/_graph_data/solo_res_5x5_to_10x10_win_rates.pkl"
                res_wr = load_pickle(data_path)

                for size_i in range(len(prog_win_rates)):
                    iterations = prog_win_rates[size_i]
                    line_lst = plt.plot(range(len(iterations)), iterations, label = str(size_i+5) + "x" + str(size_i+5))
                    color = line_lst[0].get_color()
                    wr = res_wr[size_i]
                    plt.axhline(y=wr, linestyle='--', label = str(size_i+5) + "x" + str(size_i+5) + "_ResNet", color = color)

                plt.xlabel("Recurrent Iterations")
                plt.ylabel("Win Ratio")
        
                plt.title(titulo, pad=20, fontsize = 14)

                lgd = plt.legend(bbox_to_anchor=(1,1))
                plt.gcf().canvas.draw()
                invFigure = plt.gcf().transFigure.inverted()

                lgd_pos = lgd.get_window_extent()
                lgd_coord = invFigure.transform(lgd_pos)
                lgd_xmax = lgd_coord[1, 0]

                ax_pos = plt.gca().get_window_extent()
                ax_coord = invFigure.transform(ax_pos)
                ax_xmax = ax_coord[1, 0]        

                shift = 1.1 - (lgd_xmax - ax_xmax)
                plt.gcf().tight_layout(rect=(0, 0, shift, 1))


                plt.savefig("Graphs/" + figname, dpi=300)
                plt.clf()

                return
            
            case _:
                raise Exception("Unknown plot option.")

# ------------------------------------------------------ #

def main():
    plot_maker = PlotMaker()
    plot_maker.make_plot(1)  # Make plot type 1


# ------------------------------------------------------ #

if __name__ == "__main__":
    main()
