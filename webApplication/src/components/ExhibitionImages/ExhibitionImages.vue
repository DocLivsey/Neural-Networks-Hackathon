<script>
import {defineComponent} from 'vue'

export default defineComponent({
    name: "ExhibitionImages"
})
</script>

<template>
    <div class="exhibition_image">
        <img src="/src/assets/images/flower.png">
    </div>
</template>

<style scoped>
.exhibition_image {
    width: 17.875em;
    height: 18.75em;
}
.exhibition_image > img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 0.625em;

}
</style>