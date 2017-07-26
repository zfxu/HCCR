
var drawing = false;

var context;

var offset_left = 0;
var offset_top = 0;


function start_canvas ()
{
    var scribbler = document.getElementById ("the_stage");
    context = scribbler.getContext ("2d");
    scribbler.onmousedown = function (event) {mousedown(event)};
    scribbler.onmousemove = function (event) {mousemove(event)};
    scribbler.onmouseup   = function (event) {mouseup(event)};
    for (var o = scribbler; o ; o = o.offsetParent) {
    offset_left += (o.offsetLeft - o.scrollLeft);
    offset_top  += (o.offsetTop - o.scrollTop);
    }
    draw();
}

function getPosition(evt)
{
    evt = (evt) ?  evt : ((event) ? event : null);
    var left = 0;
    var top = 0;
    var scribbler = document.getElementById("the_stage");

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
    left -= offset_left;
    top -= offset_top;

    return {x : left, y : top}; 
}

function mousedown(event)
{
    drawing = true;
    var location = getPosition(event);
    context.lineWidth = 15.0;
    context.strokeStyle="#000000";
    context.beginPath();
    context.moveTo(location.x,location.y);
}


function mousemove(event)
{
    if (!drawing) 
        return;
    var location = getPosition(event);
    context.lineTo(location.x,location.y);
    context.stroke();
}


function mouseup(event)
{
    if (!drawing) 
        return;
    mousemove(event);
    drawing = false;
}


function draw()
{

    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, 400, 400);

}


function clearCanvas()
{
    context.clearRect (0, 0, 400, 400);
    draw();

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

function test_func()
{
    document.getElementById("rec_result").innerHTML = "connecting...";
    
    var sendPackage = {"id": "1", "txt": "hello, js"};
    $.post("/process", sendPackage, function(data){
        data = JSON.parse(data);
        if(data["status"] == 1)
        {
            document.getElementById("rec_result").innerHTML = data["result"];
        }
        else
        {
            document.getElementById("rec_result").innerHTML = "failed";
        }
    });
}

onload = start_canvas;

