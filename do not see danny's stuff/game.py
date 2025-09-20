from itertools import cycle
from random import randrange
from tkinter import Canvas, Tk, messagebox, font

# Window setup
canvas_width = 800
canvas_height = 400
root = Tk()
root.title("Catch the Eggs!")

# Canvas setup
c = Canvas(root, width=canvas_width, height=canvas_height, background='deep sky blue')
c.create_rectangle(-5, canvas_height - 100, canvas_width + 5, canvas_height + 5, fill='sea green', width=0)
c.create_oval(-80, -80, 120, 120, fill='orange', width=0)
c.pack()

# Game variables
color_cycle = cycle(['light blue', 'light green', 'light pink', 'light yellow', 'light cyan'])
egg_width = 45
egg_height = 55
egg_score = 10
egg_speed = 500
egg_interval = 4000
difficulty_factor = 0.95

# Catcher setup
catcher_color = 'blue'
catcher_width = 100
catcher_height = 100
catcher_start_x = canvas_width / 2 - catcher_width / 2
catcher_start_y = canvas_height - catcher_height - 20
catcher = c.create_arc(
    catcher_start_x, catcher_start_y,
    catcher_start_x + catcher_width, catcher_start_y + catcher_height,
    start=200, extent=140, style='arc',
    outline=catcher_color, width=3
)

# Score and lives
game_font = font.nametofont('TkFixedFont')
game_font.config(size=18)

score = 0
score_text = c.create_text(10, 10, anchor='nw', font=game_font, fill='dark blue', text='Score: ' + str(score))

lives_remaining = 3
lives_text = c.create_text(canvas_width - 10, 10, anchor='ne', font=game_font, fill='dark blue', text='Lives: ' + str(lives_remaining))

# Game logic
eggs = []

def create_egg():
    x = randrange(10, canvas_width - egg_width - 10)
    y = 40
    new_egg = c.create_oval(x, y, x + egg_width, y + egg_height, fill=next(color_cycle), width=0)
    eggs.append(new_egg)
    root.after(egg_interval, create_egg)

def move_eggs():
    for egg in eggs[:]:
        (egg_x, egg_y, egg_x2, egg_y2) = c.coords(egg)
        c.move(egg, 0, 10)
        if egg_y2 > canvas_height:
            egg_dropped(egg)
    root.after(egg_speed, move_eggs)

def egg_dropped(egg):
    eggs.remove(egg)
    c.delete(egg)
    lose_a_life()

def lose_a_life():
    global lives_remaining
    lives_remaining -= 1
    c.itemconfigure(lives_text, text='Lives: ' + str(lives_remaining))
    if lives_remaining == 0:
        messagebox.showinfo('Game Over!', 'Final Score: ' + str(score))
        root.destroy()

def check_catch():
    (catcher_x1, catcher_y1, catcher_x2, catcher_y2) = c.coords(catcher)
    for egg in eggs[:]:
        (egg_x1, egg_y1, egg_x2, egg_y2) = c.coords(egg)
        if catcher_x1 < egg_x1 and egg_x2 < catcher_x2 and catcher_y2 - egg_y2 < 40:
            eggs.remove(egg)
            c.delete(egg)
            increase_score(egg_score)
    root.after(100, check_catch)

def increase_score(points):
    global score, egg_speed, egg_interval
    score += points
    egg_speed = int(egg_speed * difficulty_factor)
    egg_interval = int(egg_interval * difficulty_factor)
    c.itemconfigure(score_text, text='Score: ' + str(score))

def move_left(event):
    (x1, _, x2, _) = c.coords(catcher)
    if x1 > 0:
        c.move(catcher, -20, 0)

def move_right(event):
    (x1, _, x2, _) = c.coords(catcher)
    if x2 < canvas_width:
        c.move(catcher, 20, 0)

# Bind keys to root instead of canvas
root.bind('<Left>', move_left)
root.bind('<Right>', move_right)

# Start the game
create_egg()
move_eggs()
check_catch()


def show_crack(x1, y1, x2, y2):
    mid_x = (x1 + x2) / 2
    bottom_y = canvas_height - 10  # ground level

    # draw 3 straight "crack" lines
    crack1 = c.create_line(mid_x, bottom_y-20, mid_x, bottom_y, fill="white", width=3)
    crack2 = c.create_line(mid_x-10, bottom_y-10, mid_x+10, bottom_y, fill="white", width=2)
    crack3 = c.create_line(mid_x+10, bottom_y-10, mid_x-10, bottom_y, fill="white", width=2)

    # remove after 600ms
    root.after(600, lambda: (c.delete(crack1), c.delete(crack2), c.delete(crack3)))



def egg_dropped(egg):
    eggs.remove(egg)
    # Get egg's last position
    (x1, y1, x2, y2) = c.coords(egg)
    c.delete(egg)
    show_yolk_and_crack(x1, y1, x2, y2)
    lose_a_life()

def show_yolk_and_crack(x1, y1, x2, y2):
    mid_x = (x1 + x2) / 2
    bottom_y = canvas_height - 10  # ground level

    # Draw yolk (yellow oval)
    yolk_radius = 18
    yolk = c.create_oval(
        mid_x - yolk_radius, bottom_y - yolk_radius,
        mid_x + yolk_radius, bottom_y + yolk_radius,
        fill="yellow", outline="gold", width=2
    )

    # Draw 3 crack lines
    crack1 = c.create_line(mid_x, bottom_y-20, mid_x, bottom_y, fill="white", width=3)
    crack2 = c.create_line(mid_x-10, bottom_y-10, mid_x+10, bottom_y, fill="white", width=2)
    crack3 = c.create_line(mid_x+10, bottom_y-10, mid_x-10, bottom_y, fill="white", width=2)

    # Remove yolk and cracks after 600ms
    root.after(600, lambda: (c.delete(yolk), c.delete(crack1), c.delete(crack2), c.delete(crack3)))



root.mainloop()