<script>
import {defineComponent} from 'vue'

export default defineComponent({
    name: "ExhibitCard",
    props: {
        id: String,
        imageSrc: String,
        title: String,
        subtitle: String,
        id1: String,
        id2: String
    },
    data(){
        return {
            isPressed: false,
            id: this.id,
        }
    },
    methods: {
        pressCard(){
            this.isPressed = !this.isPressed;
            setTimeout(()=>{this.isPressed = !this.isPressed}, 200);
        }
    }
})
</script>

<template>
    <div @click="pressCard" :class="['card', {'pressed': isPressed}]">
        <RouterLink class="delete_link_style" :to="'/exhibits/' + id">
            <div class="card__image">
                <img alt="image" :src="'D:\\Coding\\HACKATHON\\model\\train_jpg\\'+imageSrc+'.jpg'">
            </div>
            <div class="card__info">
                <div class="card__ids">
                    <div class="id_1">
                        {{id1}}
                    </div>
                    <div class="id_2">
                        {{id2}}
                    </div>
                </div>
                <div class="card__name">
                    <div class="card__title">
                        {{title}}
                    </div>
                    <div class="card__subtitle">
                        {{subtitle}}
                    </div>
                </div>
            </div>
        </RouterLink>
    </div>
</template>

<style scoped>
.card {
    max-width: 17.8125em;
    min-height: 20.875em;
    border-radius: 0.625em;
    background-color: var(--first-color);
    transition: all 200ms;
    cursor: pointer;
    box-shadow: 0 0.25em 1em rgba(0, 0, 0, 0.15);
}

.card__image {
    height: 15em;
}
.card__image > img {
    border-radius: 0.625em;
}
.card__info {
    display: flex;
    flex-direction: column;
    padding: 0.625em 0.9375em;
}
.card__ids {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    font-family: var(--tertiary-font-family);
    font-size: var(--tertiary-font-size);
    color: var(--third-color);
}
.card__name {
    margin-top: 0.6875em;
    color: var(--second-color);
}
.card__title {
    font-size: var(--secondary-font-size);
    font-family: var(--secondary-font-family);
    font-weight: 600;
}
.card__subtitle {
    font-size: var(--tertiary-font-size);
    font-family: var(--secondary-font-family);
}
.pressed {
    transform: scale(0.95);
}
@media (min-width: 80em){
    .card:hover {
        transform: scale(1.04);
    }
    .pressed {
        transform: scale(1);
    }
}
</style>