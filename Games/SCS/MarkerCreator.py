from SCS_Renderer import SCS_Renderer

class MarkerCreator:

    def __init__(self):
        return
    
    def new_marker(self, option):

        match option:
            case 1:               

                renderer = SCS_Renderer()

                renderer.create_marker_from_scratch("ally_soldier", (1,2,2), "infantary", color_str="dark_green")
                renderer.add_border("green", "SCS/Images/ally_soldier.jpg")

                renderer.create_marker_from_scratch("ally_tank", (2,2,4), "mechanized", color_str="dark_green")
                renderer.add_border("green", "SCS/Images/ally_tank.jpg")

                renderer.create_marker_from_scratch("axis_soldier", (1,1,3), "infantary", color_str="dark_red")
                renderer.add_border("red", "SCS/Images/axis_soldier.jpg")

                renderer.create_marker_from_scratch("axis_tank", (4,6,1), "mechanized", color_str="dark_red")
                renderer.add_border("red", "SCS/Images/axis_tank.jpg")


                print("\nImages created!\n")
