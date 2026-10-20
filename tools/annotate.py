"""Local manual frame/identity/posture/event reviewer with transactional saves."""
import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from poseguard.image_io import read_image
from poseguard.review import ReviewStore
from poseguard.features import LABELS
from poseguard.annotations import file_sha256


def editable_people(task, saved=None):
    if saved is not None:
        return saved["people"]
    return [dict(bbox=list(p["bbox"]), person_id="", label="", identity_verified=False,
                 identity_evidence="", candidate_local_actor_id=p.get("candidate_local_actor_id", ""),
                 posture_hint=p.get("posture_hint", "unknown"), source="model_suggestion")
            for p in task.get("suggested_people", [])]


class Reviewer:
    def __init__(self, root, tasks, database):
        self.root, self.task_path = root, Path(tasks).resolve()
        self.tasks = [json.loads(s) for s in self.task_path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not self.tasks:
            raise ValueError("No review tasks")
        self.store, self.index, self.people, self.dirty = ReviewStore(database), 0, [], False
        root.title("PoseGuard Manual Review")
        root.geometry("1260x820")
        root.minsize(840,600)
        root.columnconfigure(0,weight=1)
        root.rowconfigure(1,weight=1)
        self.title = tk.StringVar()
        ttk.Label(root,textvariable=self.title).grid(row=0,column=0,sticky="w",padx=12,pady=10)
        self.canvas = tk.Canvas(root,bg="#242424",highlightthickness=0)
        self.canvas.grid(row=1,column=0,sticky="nsew")
        side = ttk.Frame(root,padding=12)
        side.grid(row=0,column=1,rowspan=3,sticky="ns")
        self.listbox = tk.Listbox(side,width=32,height=12,exportselection=False)
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>",self.select)
        self.person, self.posture, self.evidence, self.reviewer = (tk.StringVar() for _ in range(4))
        self.verified = tk.BooleanVar()
        self.hint = tk.StringVar()
        ttk.Label(side, textvariable=self.hint, wraplength=235).pack(anchor="w", pady=(6, 3))
        for label,var in (("Person",self.person),("Identity Evidence",self.evidence),("Reviewer",self.reviewer)):
            ttk.Label(side,text=label).pack(anchor="w",pady=(10,3))
            ttk.Entry(side,textvariable=var,width=32).pack(fill="x")
        ttk.Label(side,text="Posture").pack(anchor="w",pady=(10,3))
        ttk.Combobox(side,textvariable=self.posture,values=LABELS,state="readonly").pack(fill="x")
        ttk.Checkbutton(side,text="Identity Verified",variable=self.verified).pack(anchor="w",pady=10)
        for label,command in (("Apply Person",self.apply),("Delete Person",self.delete),
                              ("Complete Frame",self.save),("Fall Event",self.event),("Export",self.export)):
            ttk.Button(side,text=label,command=command).pack(fill="x",pady=4)
        nav = ttk.Frame(root,padding=10)
        nav.grid(row=2,column=0,sticky="ew")
        ttk.Button(nav,text="<",command=lambda:self.move(-1)).pack(side="left")
        ttk.Button(nav,text=">",command=lambda:self.move(1)).pack(side="left",padx=8)
        self.status = tk.StringVar()
        ttk.Label(nav,textvariable=self.status).pack(side="left",padx=8)
        self.canvas.bind("<Configure>",lambda e:self.draw())
        self.canvas.bind("<ButtonPress-1>",self.press)
        self.canvas.bind("<B1-Motion>",self.drag)
        self.canvas.bind("<ButtonRelease-1>",self.release)
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.load()

    def load(self):
        self.task = self.tasks[self.index]
        image_path = self.task_path.parent/self.task["image"]
        image_hash = file_sha256(image_path)
        self.image = read_image(image_path)
        if self.image is None:
            raise ValueError("Unreadable task image")
        if file_sha256(image_path) != image_hash:
            raise ValueError("Task image changed during load; reload the task")
        self.loaded_image_sha256 = image_hash
        saved = self.store.get(self.task)
        self.people = editable_people(self.task, saved)
        self.dirty = False
        self.title.set(f'{self.task["video_id"]} / {self.task.get("frame_id",self.task.get("source_frame"))} / {self.task["timestamp"]:.3f}s')
        self.status.set(f'{self.index+1}/{len(self.tasks)}  '+("Complete" if saved else "Pending Human Review"))
        self.refresh()

    def refresh(self):
        self.listbox.delete(0,"end")
        for p in self.people:
            self.listbox.insert("end",f'{p.get("person_id") or p.get("candidate_local_actor_id", "?")}  {p.get("label") or "?"}')
        self.draw()

    def draw(self):
        if not hasattr(self,"image"):
            return
        h,w = self.image.shape[:2]
        self.scale = min(max(self.canvas.winfo_width(),1)/w,max(self.canvas.winfo_height(),1)/h)
        resized = cv2.resize(self.image,(max(1,int(w*self.scale)),max(1,int(h*self.scale))))
        rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        data = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode()+rgb.tobytes()
        self.photo = tk.PhotoImage(data=data,format="PPM")
        self.canvas.delete("all")
        self.canvas.create_image(0,0,image=self.photo,anchor="nw")
        for p in self.people:
            b = [v*self.scale for v in p["bbox"]]
            color = "#f0c04a" if p.get("source") == "model_suggestion" else "#35daab"
            self.canvas.create_rectangle(*b,outline=color,width=2)
            self.canvas.create_text(b[0]+3,b[1]+3,text=p.get("person_id") or "?",anchor="nw",fill="white")

    def point(self,e):
        h,w = self.image.shape[:2]
        return max(0,min(w,e.x/self.scale)),max(0,min(h,e.y/self.scale))

    def press(self,e):
        self.origin = self.point(e)
        self.rubber = self.canvas.create_rectangle(e.x,e.y,e.x,e.y,outline="#f0c04a",width=2)

    def drag(self,e):
        self.canvas.coords(self.rubber,self.origin[0]*self.scale,self.origin[1]*self.scale,e.x,e.y)

    def release(self,e):
        end = self.point(e)
        box = [min(self.origin[0],end[0]),min(self.origin[1],end[1]),max(self.origin[0],end[0]),max(self.origin[1],end[1])]
        if box[2]-box[0] < 2 or box[3]-box[1] < 2:
            return
        self.people.append(dict(bbox=box,person_id="",label="",identity_verified=False,identity_evidence=""))
        self.dirty = True
        self.refresh()
        self.listbox.selection_set(len(self.people)-1)
        self.select()

    def select(self,e=None):
        ids = self.listbox.curselection()
        if ids:
            p = self.people[ids[0]]
            self.person.set(p["person_id"])
            self.posture.set(p["label"])
            self.evidence.set(p.get("identity_evidence",""))
            self.verified.set(p.get("identity_verified",False))
            self.hint.set(f'Posture hint: {p.get("posture_hint", "none")}\nLocal track: {p.get("candidate_local_actor_id", "none")}')

    def apply(self):
        ids = self.listbox.curselection()
        if ids:
            self.people[ids[0]].update(person_id=self.person.get().strip(),label=self.posture.get(),
                                      identity_evidence=self.evidence.get().strip(),identity_verified=self.verified.get(),
                                      source="manual")
            self.dirty = True
            self.refresh()

    def delete(self):
        ids = self.listbox.curselection()
        if ids:
            self.people.pop(ids[0]); self.dirty=True; self.refresh()

    def save(self):
        try:
            self.store.save(self.task,self.people,self.reviewer.get(),image_root=self.task_path.parent,
                            expected_image_sha256=self.loaded_image_sha256)
            self.dirty=False
            self.status.set(f'{self.index+1}/{len(self.tasks)}  Complete')
        except Exception as exc:
            messagebox.showerror("Frame",str(exc))

    def move(self,delta):
        if self.dirty:
            self.save()
            if self.dirty:
                return
        self.index=max(0,min(len(self.tasks)-1,self.index+delta)); self.load()

    def event(self):
        try:
            start=simpledialog.askfloat("Fall Event","Start (seconds)",initialvalue=self.task["timestamp"])
            if start is None: return
            end=simpledialog.askfloat("Fall Event","End (seconds)",initialvalue=start)
            if end is None: return
            self.store.save_event(self.task["video_id"],self.person.get().strip(),start,end,self.reviewer.get())
        except Exception as exc:
            messagebox.showerror("Event",str(exc))

    def export(self):
        path=simpledialog.askstring("Export","Output directory")
        if path:
            try:
                result=self.store.export(self.task_path,path)
                messagebox.showinfo("Export",f'{result["frames"]} frames / {result["people_instances"]} people')
            except Exception as exc:
                messagebox.showerror("Export",str(exc))

    def close(self):
        if self.dirty:
            self.save()
            if self.dirty: return
        self.store.connection.close(); self.root.destroy()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    suggestions = Path("data/urfall-review-draft-suggestions/draft-review.jsonl")
    parser.add_argument("--tasks", default=str(suggestions) if suggestions.is_file()
                        else "data/urfall-review-dense-rgb/review-tasks.jsonl")
    parser.add_argument("--database",default="data/manual-review.sqlite")
    args=parser.parse_args()
    root=tk.Tk()
    app=Reviewer(root,args.tasks,args.database)
    root.mainloop()
