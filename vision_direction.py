import matplotlib.pyplot as plt

def find_perpendicular_point(A, B, C):
    """
    Finds the point on line segment AB where a perpendicular from point C would intersect.
    
    Parameters:
    A (tuple): Coordinates of point A (x_A, y_A).
    B (tuple): Coordinates of point B (x_B, y_B).
    C (tuple): Coordinates of point C (x_C, y_C).
    
    Returns:
    tuple: Coordinates of the perpendicular intersection point on AB.
    """
    # Unpack the coordinates
    x_A, y_A = A
    x_B, y_B = B
    x_C, y_C = C
    
    # Calculate the direction vector from A to B
    AB_x = x_B - x_A
    AB_y = y_B - y_A
    
    # Calculate the vector from A to C
    AC_x = x_C - x_A
    AC_y = y_C - y_A
    
    # Projection formula to find the scalar t for point D on AB
    t = (AC_x * AB_x + AC_y * AB_y) / (AB_x ** 2 + AB_y ** 2)
    
    # Calculate the coordinates of the perpendicular point D
    x_D = x_A + t * AB_x
    y_D = y_A + t * AB_y
    
    return (x_D, y_D)

# Example coordinates for points A, B, and C
A = (1, 0)
B = (4, 5)
C = (2, 5)
perpendicular_point = find_perpendicular_point(A, B, C)
print("The perpendicular point on line AB from point C is:", perpendicular_point)










############################# FOR PLOTTING ONLY
# Plotting
plt.figure(figsize=(8, 6))
plt.plot([A[0], B[0]], [A[1], B[1]], 'r-', label="Line Segment AB")  # Line AB
plt.plot([C[0], perpendicular_point[0]], [C[1], perpendicular_point[1]], 'b--', label="Perpendicular from C")  # Perpendicular line

# Plot points
plt.scatter(*A, color='black', label="Point A")
plt.scatter(*B, color='black', label="Point B")
plt.scatter(*C, color='black', label="Point C")
plt.scatter(*perpendicular_point, color='purple', label="Point D (Intersection)", marker='x')

# Annotate points
plt.text(A[0], A[1], '  A', color="black", verticalalignment='bottom')
plt.text(B[0], B[1], '  B', color="black", verticalalignment='bottom')
plt.text(C[0], C[1], '  C', color="black", verticalalignment='top')
plt.text(perpendicular_point[0], perpendicular_point[1], f'  D ({perpendicular_point[0]:.2f},{perpendicular_point[1]:.2f})', color="purple", verticalalignment='top')

# Annotation for 90-degree angle verification
plt.annotate("90°", xy=(perpendicular_point[0], perpendicular_point[1]), xytext=(perpendicular_point[0] + 0.3, perpendicular_point[1] + 0.5),
             arrowprops=dict(arrowstyle="->", color='blue', lw=1.5),
             fontsize=12, color="blue")

# Set plot properties
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)
plt.legend()
plt.title("Perpendicular Intersection of Point C on Line Segment AB")
plt.grid(False)  # Disable grid
plt.show()
