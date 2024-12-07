window.onload = () => {
    $('#sendbutton').click(() => {
        imagebox = $('#imagebox')
        input = $('#imageinput')[0]
        if(input.files && input.files[0])
        {
            let formData = new FormData();
            formData.append('image' , input.files[0]);
            $.ajax({
                url: "http://localhost:5000/detectObject", 
                type:"POST",
                data: formData,
                cache: false,
                processData:false,
                contentType:false,
                error: function(data){
                    console.log("upload error" , data);
                    console.log(data.getAllResponseHeaders());
                },
                success: function(data){
                    console.log(data);
                    
                    // Display image
                    bytestring = data['status']
                    image = bytestring.split('\'')[1]
                    imagebox.attr('src' , 'data:image/jpeg;base64,'+image)
                    
                    // Display detected objects
                    let objectsContainer = $('#detected-objects-container')
                    objectsContainer.empty()  // Clear previous results
                    
                    if(data['objects'] && data['objects'].length > 0) {
                        // Consolidate objects with same name
                        const consolidatedObjects = data['objects'].reduce((acc, obj) => {
                            const existingObj = acc.find(item => item.name === obj.name);
                            if (existingObj) {
                                existingObj.confidence = Math.max(existingObj.confidence, obj.confidence);
                            } else {
                                acc.push({...obj});
                            }
                            return acc;
                        }, []);

                        // Sort by confidence in descending order
                        consolidatedObjects.sort((a, b) => b.confidence - a.confidence);

                        let objectsHtml = `
                            <h2 class="detected-objects-title">Detected Objects</h2>
                            <div class="object-detection-results">
                        `;
                        
                        consolidatedObjects.forEach(obj => {
                            objectsHtml += `
                                <div class="object-item">
                                    <span class="object-name">${obj.name}</span>
                                    <span class="object-confidence">${(obj.confidence * 100).toFixed(2)}%</span>
                                </div>
                            `
                        });
                        
                        objectsHtml += '</div>';
                        objectsContainer.html(objectsHtml)
                    } else {
                        objectsContainer.html('<p class="no-objects">No objects detected</p>')
                    }
                }
            });
        }
    });
};

function readUrl(input){
    imagebox = $('#imagebox')
    console.log("evoked readUrl")
    if(input.files && input.files[0]){
        let reader = new FileReader();
        reader.onload = function(e){
            imagebox.attr('src',e.target.result); 
        }
        reader.readAsDataURL(input.files[0]);
    }
}