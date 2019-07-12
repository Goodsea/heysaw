$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
	
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
				//$('#upload-file').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
            },
        });
    });

});
(function() {
    let showToastButton = document.querySelector('#yardim');
    showToastButton.addEventListener('click', function() {
		document.querySelector('#yardimicerik').innerHTML = "Hello, <b>heysaw</b> is an artificial intelligence model developed to help doctors diagnose retinal diseases. <br><br> To use it, simply click the file selection button on the left side of the screen to upload the Optical Coherence Tomography (OCT) image. Then <b> heysaw </b> detects diseases that are present / likely to be present in the retina for you. <b> heysaw </b> has 98.9% accuracy in unprecedented test data (1K images) and is open source. You can access the open-sourced codes by clicking <a href='https://github.com/Goodsea/heysaw' target='_blank'>here</a>.";
    });
}());
