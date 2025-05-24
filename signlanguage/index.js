const express = require('express');
const cors = require('cors');
const app = express();
const bodyparser = require('body-parser');

app.use(bodyparser.json());
app.use(express.urlencoded({extended:true}));
app.use(cors());
app.use(express.json());
app.use(express.static("public"));

app.get("/",(req,res)=>{
    res.render("index2.ejs");
})

app.get("/lang",(req,res)=>{
    res.render("language3.ejs");
})

app.post('/get-prediction',(req, res) => {
    var l = req.body;
    console.log(l);
    try {
        res.render("cam.ejs",{
            lang:l.language
        });
    } catch (err) {
        res.status(500).send('Python server error');
    }
});

app.post("/admin",(req,res)=>{
    res.render("admin.ejs")
})

app.listen(3000, () => console.log('Node.js server running on http://localhost:3000'));
