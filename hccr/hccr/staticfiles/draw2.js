
function Handwriting(id) {
    this.canvas = document.getElementById(id);
    this.ctx = this.canvas.getContext("2d");
    this.ctx.fillStyle = "#ffffff";
    this.ctx.fillRect(0, 0, 400, 400);
    this.ctx.strokeStyle = "#000000";
    var _this = this;
    this.canvas.onmousedown = function (e) { _this.downEvent(e)};
    this.canvas.onmousemove = function (e) { _this.moveEvent(e)};
    this.canvas.onmouseup = function (e) { _this.upEvent(e)};
    this.canvas.onmouseout = function (e) { _this.upEvent(e)};
    this.moveFlag = false;
    this.upof = {};
    this.radius = 0;
    this.has = [];
    this.lineMax = 30;
    this.lineMin = 3;
    this.linePressure = 1;
    this.smoothness = 80;
    this.offset_left = 0;
    this.offset_top = 0;

    for (var o = this.canvas; o ; o = o.offsetParent) {
        this.offset_left += (o.offsetLeft - o.scrollLeft);
        this.offset_top  += (o.offsetTop - o.scrollTop);
    }
}

Handwriting.prototype.getPosition = function(evt)
{
    evt = (evt) ?  evt : ((event) ? event : null);
    var left = 0;
    var top = 0;

    if (evt.pageX) {
    left = evt.pageX;
    top  = evt.pageY;
    } else if (document.documentElement.scrollLeft) {
    left = evt.clientX + document.documentElement.scrollLeft;
    top  = evt.clientY + document.documentElement.scrollTop;
    } else  {
    left = evt.clientX + document.body.scrollLeft;
    top  = evt.clientY + document.body.scrollTop;
    }
    left -= this.offset_left;
    top -= this.offset_top;

    return {x : left, y : top}; 
}

Handwriting.prototype.clear = function () {
    this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
}

Handwriting.prototype.downEvent = function (e) {
    this.moveFlag = true;
    this.has = [];

    this.upof = this.getPosition(e);
    //this.ctx.beginPath();
    //this.ctx.moveTo(this.upof.x, this.upof.y);
}

Handwriting.prototype.moveEvent = function (e) {
    if (!this.moveFlag)
        return;
    var of = this.getPosition(e);
    var up = this.upof;
    var ur = this.radius;
    this.has.unshift({time:new Date().getTime() ,dis:this.distance(up,of)});
    var dis = 0;
    var time = 0;
    for (var n = 0; n < this.has.length-1; n++) {
        dis += this.has[n].dis;
        time += this.has[n].time-this.has[n+1].time;
        if (dis>this.smoothness)
            break;
    }
    var or = Math.min(time/dis*this.linePressure+this.lineMin , this.lineMax) / 2;
    this.radius = or;
    this.upof = of;
    if (this.has.length<=4)
        return;
    var len = Math.round(this.has[0].dis/2)+1;
    for (var i = 0; i < len; i++) {
        var x = up.x + (of.x-up.x)/len*i;
        var y = up.y + (of.y-up.y)/len*i;
        var r = ur + (or-ur)/len*i;
        this.ctx.beginPath();
        this.ctx.arc(x,y,r,0,2*Math.PI,true);
        this.ctx.fillStyle = "#000000";
        this.ctx.fill();
        //this.ctx.lineTo(up.x,up.y);
        //this.ctx.stroke();
    }
}

Handwriting.prototype.upEvent = function (e) {
    this.moveFlag = false;
}

Handwriting.prototype.getXY = function (e)
{
    return {
        x : e.clientX - this.canvas.offsetLeft + (document.body.scrollLeft || document.documentElement.scrollLeft),
        y : e.clientY - this.canvas.offsetTop  + (document.body.scrollTop || document.documentElement.scrollTop)
    }
}

Handwriting.prototype.distance = function (a,b)
{
    var x = b.x-a.x , y = b.y-a.y;
    return Math.sqrt(x*x+y*y);
}

function clearCanvas()
{
    hw.ctx.fillStyle = '#ffffff';
    hw.ctx.fillRect(0, 0, 400, 400);

    document.getElementById("connection_info").innerHTML = "&nbsp;";
    for(var i = 0; i < 5; i++)
    {
        var str1 = "#table_body tr:nth-child(" + (i+1) + ") td:nth-child(2)";
        var str2 = "#table_body tr:nth-child(" + (i+1) + ") td:nth-child(3)";
        document.querySelector(str1).innerHTML = "";
        document.querySelector(str2).innerHTML = "";
    }
}


function processImg()
{
    document.getElementById("connection_info").innerHTML = "connecting...";
    
    var scribbler = document.getElementById ("the_stage");
    var imageData =  scribbler.toDataURL('image/png');
    var dataTemp = imageData.substr(22);  

    var sendPackage = {"id": "1", "txt": dataTemp};
    $.post("/process", sendPackage, function(data){
        data = JSON.parse(data);

        if(data["status"] == 1)
        {
            document.getElementById("connection_info").innerHTML = "&nbsp;";
            for(var i = 0; i < 5; i++)
            {
                var str1 = "#table_body tr:nth-child(" + (i+1) + ") td:nth-child(2)";
                var str2 = "#table_body tr:nth-child(" + (i+1) + ") td:nth-child(3)";
                document.querySelector(str1).innerHTML = data["char"][i];
                document.querySelector(str2).innerHTML = data["val"][i].toFixed(4);
            }
        }
        else
        {
            document.getElementById("rec_result").innerHTML = "failed";
        }
    });

    document.getElementById("connection_info").innerHTML = "&nbsp;";
}  


var hw = new Handwriting("the_stage");
hw.lineMax = 15;
hw.lineMin = 5;
hw.linePressure = 1;
hw.smoothness = 100;