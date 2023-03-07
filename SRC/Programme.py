from pathlib import Path
from tkinter import Label, StringVar,Tk,Canvas,Text,Button,PhotoImage
from tkinter.constants import END,WORD
from sklearn import svm
import seaborn as sns
import pickle

#chemin d'accès aux assets
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()
window.title("TP4 LOAN")
width = 600 
height = 300
 
screen_width = window.winfo_screenwidth()  
screen_height = window.winfo_screenheight() 

x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
 
window.geometry('%dx%d+%d+%d' % (width, height, x, y))
window.resizable(False, False)
window.config(cursor="dot")

result=StringVar()
result.set("------")

#import de notre modele SVM
with open('Loan_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
    

def predire():
    ok=False
    try:
        inputs=[[float(LA.get(0.0,END)),float(CH.get(0.0,END))]]
        ok=True
    except:
        result.set('Entrer des bonnes Valeurs')
        ok=False
    if ok:
        result.set(model.predict(inputs))
        
    
#arrière plan
canvas = Canvas(
    window,
    bg="#004c7f",
    height=300,
    width=600,
    bd=0,
    highlightthickness=0,
    relief="ridge",
)
canvas.place(x = 0, y = 0)

bg_image = PhotoImage(
    file=relative_to_assets("bg.png"))
bg = canvas.create_image(
    300,
    150,
    image=bg_image
    )

LA = Text(
    canvas,
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    font=("sergio",10,"bold"),
    wrap=WORD,
    cursor="dot",

)
LA.place(
    x=155,
    y=113,
    width=110,
    height=20
)
CH = Text(
    canvas,
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    font=("sergio",10,"bold"),
    wrap=WORD,
    cursor="dot",

)
CH.place(
    x=325,
    y=113,
    width=110,
    height=20
)

button_image = PhotoImage(
    file=relative_to_assets("btn.png"))
button_1 = Button(
    canvas,
    command=predire,
    image=button_image,
    borderwidth=0,
    highlightthickness=0,
    relief="flat",
    activebackground="#9a2b2b",
    bg="#9a2b2b"
)
button_1.place(
    x=266,
    y=160,
    width=69,
    height=26
)

resultat=Label(
    canvas,
    textvariable=result,
    fg="red",
    bg="white",
    font=('sergio',13,'bold')
)
resultat.place(
        x=150,
        y=230,
        width=300
)

window.mainloop()
